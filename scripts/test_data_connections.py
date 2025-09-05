#!/usr/bin/env python3
"""
Script de test des connexions de données pour Towerco AIOps
"""

import asyncio
import sys
import json
from typing import Dict, Any, List

# Ajouter le chemin du backend
sys.path.append('../backend')

from core.database import DatabaseManager
from workers.connectors import (
    OSS_NetworkConnector,
    ITSM_ServiceNowConnector,
    Energy_IoTConnector,
    SNMPConnector,
    RESTConnector
)


class ConnectionTester:
    """Testeur de connexions de données"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.results = {}
    
    async def initialize(self):
        """Initialiser le testeur"""
        await self.db.initialize()
    
    async def test_all_sources(self):
        """Tester toutes les sources configurées"""
        sources = await self.load_data_sources()
        
        if not sources:
            print("❌ Aucune source de données configurée")
            return
        
        print(f"🧪 Test de {len(sources)} sources de données...")
        print("=" * 80)
        
        for source in sources:
            await self.test_source(source)
        
        self.print_summary()
    
    async def test_source(self, source: Dict[str, Any]):
        """Tester une source spécifique"""
        source_id = source['source_id']
        source_name = source['source_name']
        source_type = source['source_type']
        
        print(f"\n🔍 Test de {source_name} ({source_type})...")
        
        try:
            # Créer le connecteur
            connector = await self.create_connector(source)
            if not connector:
                self.results[source_id] = {
                    'status': 'error',
                    'message': 'Impossible de créer le connecteur'
                }
                return
            
            # Test de connexion
            start_time = asyncio.get_event_loop().time()
            await connector.test_connection()
            connection_time = asyncio.get_event_loop().time() - start_time
            
            # Test de collecte de données
            start_time = asyncio.get_event_loop().time()
            data = await connector.fetch_data()
            fetch_time = asyncio.get_event_loop().time() - start_time
            
            # Vérifier la santé
            health = await connector.health_check()
            
            # Nettoyer
            await connector.disconnect()
            
            # Enregistrer les résultats
            self.results[source_id] = {
                'status': 'success',
                'connection_time': round(connection_time, 2),
                'fetch_time': round(fetch_time, 2),
                'data_count': len(data) if data else 0,
                'health': health,
                'message': f"✅ Connexion réussie - {len(data) if data else 0} éléments collectés"
            }
            
            print(f"   ✅ Connexion réussie ({connection_time:.2f}s)")
            print(f"   📊 {len(data) if data else 0} éléments collectés ({fetch_time:.2f}s)")
            
        except Exception as e:
            self.results[source_id] = {
                'status': 'error',
                'message': f"❌ Erreur: {str(e)}"
            }
            print(f"   ❌ Erreur: {str(e)}")
    
    async def create_connector(self, source: Dict[str, Any]):
        """Créer un connecteur pour une source"""
        source_type = source['source_type']
        
        try:
            if source_type == 'oss_network':
                return OSS_NetworkConnector(
                    data_source=source,
                    message_producer=None,
                    quality_validator=None
                )
            elif source_type == 'itsm_servicenow':
                return ITSM_ServiceNowConnector(
                    data_source=source,
                    message_producer=None,
                    quality_validator=None
                )
            elif source_type == 'energy_iot':
                return Energy_IoTConnector(
                    data_source=source,
                    message_producer=None,
                    quality_validator=None
                )
            elif source_type == 'snmp':
                return SNMPConnector(
                    data_source=source,
                    message_producer=None,
                    quality_validator=None
                )
            elif source_type == 'rest':
                return RESTConnector(
                    data_source=source,
                    message_producer=None,
                    quality_validator=None
                )
            else:
                print(f"   ❌ Type de connecteur non supporté: {source_type}")
                return None
                
        except Exception as e:
            print(f"   ❌ Erreur lors de la création du connecteur: {e}")
            return None
    
    async def load_data_sources(self) -> List[Dict[str, Any]]:
        """Charger les sources de données"""
        query = """
        SELECT source_id, source_name, source_type, connection_config, 
               mapping_rules, enabled
        FROM data_sources
        WHERE enabled = true
        ORDER BY source_name
        """
        
        return await self.db.execute_query(query)
    
    def print_summary(self):
        """Afficher le résumé des tests"""
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 80)
        
        total = len(self.results)
        successful = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed = total - successful
        
        print(f"Total des sources: {total}")
        print(f"✅ Succès: {successful}")
        print(f"❌ Échecs: {failed}")
        
        if successful > 0:
            avg_connection_time = sum(
                r['connection_time'] for r in self.results.values() 
                if r['status'] == 'success'
            ) / successful
            print(f"⏱️  Temps de connexion moyen: {avg_connection_time:.2f}s")
        
        print("\n📋 Détails par source:")
        print("-" * 80)
        
        for source_id, result in self.results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} {source_id}: {result['message']}")
            
            if result['status'] == 'success':
                print(f"   ⏱️  Connexion: {result['connection_time']}s")
                print(f"   📊 Données: {result['data_count']} éléments")
    
    async def test_specific_source(self, source_id: str):
        """Tester une source spécifique"""
        query = "SELECT * FROM data_sources WHERE source_id = $1"
        source = await self.db.execute_query_one(query, source_id)
        
        if not source:
            print(f"❌ Source '{source_id}' non trouvée")
            return
        
        print(f"🧪 Test de la source '{source['source_name']}'...")
        await self.test_source(source)
    
    async def generate_report(self, output_file: str = "connection_test_report.json"):
        """Générer un rapport détaillé"""
        report = {
            'timestamp': asyncio.get_event_loop().time(),
            'total_sources': len(self.results),
            'successful_sources': sum(1 for r in self.results.values() if r['status'] == 'success'),
            'failed_sources': sum(1 for r in self.results.values() if r['status'] == 'error'),
            'sources': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Rapport généré: {output_file}")


async def main():
    """Fonction principale"""
    if len(sys.argv) < 2:
        print("Usage: python test_data_connections.py <command> [options]")
        print("\nCommandes disponibles:")
        print("  all                    - Tester toutes les sources")
        print("  source <source_id>     - Tester une source spécifique")
        print("  report                 - Générer un rapport détaillé")
        return
    
    command = sys.argv[1]
    tester = ConnectionTester()
    
    try:
        await tester.initialize()
        
        if command == "all":
            await tester.test_all_sources()
        elif command == "source":
            if len(sys.argv) < 3:
                print("❌ Usage: python test_data_connections.py source <source_id>")
                return
            source_id = sys.argv[2]
            await tester.test_specific_source(source_id)
        elif command == "report":
            await tester.test_all_sources()
            await tester.generate_report()
        else:
            print(f"❌ Commande inconnue: {command}")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    finally:
        await tester.db.close()


if __name__ == "__main__":
    asyncio.run(main())
