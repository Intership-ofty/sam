#!/usr/bin/env python3
"""
Script de démarrage des workers de données pour Towerco AIOps
"""

import asyncio
import sys
import signal
import logging
from typing import List, Dict, Any
from datetime import datetime

# Ajouter le chemin du backend
sys.path.append('../backend')

from core.database import DatabaseManager
from core.messaging import MessageProducer
from workers.connectors import (
    OSS_NetworkConnector,
    ITSM_ServiceNowConnector,
    Energy_IoTConnector,
    SNMPConnector,
    RESTConnector
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataWorkerManager:
    """Gestionnaire des workers de données"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.producer = MessageProducer()
        self.connectors = {}
        self.running = False
        
    async def initialize(self):
        """Initialiser le gestionnaire"""
        await self.db.initialize()
        await self.producer.initialize()
        logger.info("DataWorkerManager initialisé")
    
    async def load_data_sources(self) -> List[Dict[str, Any]]:
        """Charger les sources de données depuis la base"""
        query = """
        SELECT source_id, source_name, source_type, connection_config, 
               mapping_rules, enabled
        FROM data_sources
        WHERE enabled = true
        ORDER BY source_name
        """
        
        sources = await self.db.execute_query(query)
        logger.info(f"Chargé {len(sources)} sources de données")
        return sources
    
    async def create_connector(self, source: Dict[str, Any]):
        """Créer un connecteur pour une source"""
        source_type = source['source_type']
        source_id = source['source_id']
        
        try:
            if source_type == 'oss_network':
                connector = OSS_NetworkConnector(
                    data_source=source,
                    message_producer=self.producer,
                    quality_validator=None
                )
            elif source_type == 'itsm_servicenow':
                connector = ITSM_ServiceNowConnector(
                    data_source=source,
                    message_producer=self.producer,
                    quality_validator=None
                )
            elif source_type == 'energy_iot':
                connector = Energy_IoTConnector(
                    data_source=source,
                    message_producer=self.producer,
                    quality_validator=None
                )
            elif source_type == 'snmp':
                connector = SNMPConnector(
                    data_source=source,
                    message_producer=self.producer,
                    quality_validator=None
                )
            elif source_type == 'rest':
                connector = RESTConnector(
                    data_source=source,
                    message_producer=self.producer,
                    quality_validator=None
                )
            else:
                logger.error(f"Type de connecteur non supporté: {source_type}")
                return None
            
            # Initialiser le connecteur
            await connector.initialize()
            self.connectors[source_id] = connector
            logger.info(f"Connecteur créé pour {source['source_name']} ({source_type})")
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du connecteur {source_id}: {e}")
    
    async def start_connector(self, source_id: str, polling_interval: int = 300):
        """Démarrer un connecteur avec polling"""
        connector = self.connectors.get(source_id)
        if not connector:
            logger.error(f"Connecteur {source_id} non trouvé")
            return
        
        source_name = connector.data_source['source_name']
        logger.info(f"Démarrage du polling pour {source_name} (intervalle: {polling_interval}s)")
        
        while self.running:
            try:
                # Collecter les données
                data = await connector.fetch_data()
                
                if data:
                    # Envoyer les données via Kafka
                    for item in data:
                        await self.producer.send_message(
                            topic='data-ingestion',
                            message=item
                        )
                    
                    logger.info(f"Collecté {len(data)} éléments depuis {source_name}")
                else:
                    logger.debug(f"Aucune donnée collectée depuis {source_name}")
                
                # Attendre avant la prochaine collecte
                await asyncio.sleep(polling_interval)
                
            except Exception as e:
                logger.error(f"Erreur lors de la collecte depuis {source_name}: {e}")
                await asyncio.sleep(60)  # Attendre 1 minute en cas d'erreur
    
    async def start_all_connectors(self):
        """Démarrer tous les connecteurs"""
        sources = await self.load_data_sources()
        
        if not sources:
            logger.warning("Aucune source de données configurée")
            return
        
        # Créer tous les connecteurs
        for source in sources:
            await self.create_connector(source)
        
        # Démarrer le polling pour chaque connecteur
        tasks = []
        for source in sources:
            source_id = source['source_id']
            polling_interval = source.get('polling_interval', 300)
            
            task = asyncio.create_task(
                self.start_connector(source_id, polling_interval)
            )
            tasks.append(task)
        
        # Attendre que tous les tasks se terminent
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all_connectors(self):
        """Arrêter tous les connecteurs"""
        logger.info("Arrêt de tous les connecteurs...")
        
        for source_id, connector in self.connectors.items():
            try:
                await connector.disconnect()
                logger.info(f"Connecteur {source_id} arrêté")
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt du connecteur {source_id}: {e}")
        
        self.connectors.clear()
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifier la santé de tous les connecteurs"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_connectors': len(self.connectors),
            'healthy_connectors': 0,
            'unhealthy_connectors': 0,
            'connectors': {}
        }
        
        for source_id, connector in self.connectors.items():
            try:
                health = await connector.health_check()
                health_status['connectors'][source_id] = health
                
                if health['healthy']:
                    health_status['healthy_connectors'] += 1
                else:
                    health_status['unhealthy_connectors'] += 1
                    
            except Exception as e:
                health_status['connectors'][source_id] = {
                    'healthy': False,
                    'error': str(e)
                }
                health_status['unhealthy_connectors'] += 1
        
        return health_status
    
    async def run(self):
        """Exécuter le gestionnaire"""
        self.running = True
        logger.info("Démarrage du DataWorkerManager...")
        
        try:
            await self.initialize()
            await self.start_all_connectors()
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur fatale: {e}")
        finally:
            await self.stop_all_connectors()
            await self.producer.close()
            await self.db.close()
            logger.info("DataWorkerManager arrêté")


async def main():
    """Fonction principale"""
    manager = DataWorkerManager()
    
    # Gestionnaire de signaux pour arrêt propre
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} reçu, arrêt en cours...")
        manager.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Démarrer le gestionnaire
    await manager.run()


if __name__ == "__main__":
    asyncio.run(main())
