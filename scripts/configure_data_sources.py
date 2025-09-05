#!/usr/bin/env python3
"""
Script de configuration des sources de données pour Towerco AIOps
Permet d'ajouter facilement de nouvelles sources de données
"""

import asyncio
import json
import sys
from typing import Dict, Any, List
from datetime import datetime

# Ajouter le chemin du backend
sys.path.append('../backend')

from core.database import DatabaseManager


class DataSourceConfigurator:
    """Configureur de sources de données"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    async def add_oss_source(self, config: Dict[str, Any]) -> str:
        """Ajouter une source OSS"""
        source_id = config.get('source_id', f"oss-{config['oss_type']}-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        source_config = {
            "source_id": source_id,
            "source_name": config.get('source_name', f"{config['oss_type'].title()} OSS"),
            "source_type": "oss_network",
            "connection_config": {
                "base_url": config['base_url'],
                "oss_type": config['oss_type'],
                "api_version": config.get('api_version', 'v1'),
                "authentication": config['authentication'],
                "polling_endpoints": config.get('polling_endpoints', []),
                "technologies": config.get('technologies', ['4G', '5G']),
                "measurement_types": config.get('measurement_types', []),
                "site_filter": config.get('site_filter'),
                "max_alarms": config.get('max_alarms', 1000)
            },
            "mapping_rules": {
                "tenant_mapping": config.get('tenant_mapping', 'default-tenant'),
                "site_mapping": config.get('site_mapping', {})
            },
            "enabled": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await self._insert_source(source_config)
        return source_id
    
    async def add_servicenow_source(self, config: Dict[str, Any]) -> str:
        """Ajouter une source ServiceNow"""
        source_id = config.get('source_id', f"servicenow-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        source_config = {
            "source_id": source_id,
            "source_name": config.get('source_name', 'ServiceNow Instance'),
            "source_type": "itsm_servicenow",
            "connection_config": {
                "instance_url": config['instance_url'],
                "api_version": config.get('api_version', 'v1'),
                "authentication": config['authentication'],
                "tables": config.get('tables', ['incident', 'sc_request', 'change_request']),
                "sync_mode": config.get('sync_mode', 'incremental'),
                "max_records": config.get('max_records', 1000),
                "table_filters": config.get('table_filters', {})
            },
            "mapping_rules": {
                "tenant_mapping": config.get('tenant_mapping', 'default-tenant'),
                "site_field_mapping": config.get('site_field_mapping')
            },
            "enabled": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await self._insert_source(source_config)
        return source_id
    
    async def add_iot_source(self, config: Dict[str, Any]) -> str:
        """Ajouter une source IoT"""
        source_id = config.get('source_id', f"iot-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        source_config = {
            "source_id": source_id,
            "source_name": config.get('source_name', 'IoT Energy System'),
            "source_type": "energy_iot",
            "connection_config": {
                "base_url": config['base_url'],
                "authentication": config['authentication'],
                "sensor_types": config.get('sensor_types', ['power', 'voltage', 'current']),
                "polling_interval": config.get('polling_interval', 60),
                "batch_size": config.get('batch_size', 100)
            },
            "mapping_rules": {
                "tenant_mapping": config.get('tenant_mapping', 'default-tenant'),
                "site_mapping": config.get('site_mapping', {})
            },
            "enabled": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await self._insert_source(source_config)
        return source_id
    
    async def add_snmp_source(self, config: Dict[str, Any]) -> str:
        """Ajouter une source SNMP"""
        source_id = config.get('source_id', f"snmp-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        source_config = {
            "source_id": source_id,
            "source_name": config.get('source_name', 'SNMP Devices'),
            "source_type": "snmp",
            "connection_config": {
                "targets": config['targets'],
                "oids": config.get('oids', []),
                "polling_interval": config.get('polling_interval', 300)
            },
            "mapping_rules": {
                "tenant_mapping": config.get('tenant_mapping', 'default-tenant'),
                "site_mapping": config.get('site_mapping', {})
            },
            "enabled": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await self._insert_source(source_config)
        return source_id
    
    async def _insert_source(self, source_config: Dict[str, Any]):
        """Insérer une source dans la base de données"""
        query = """
        INSERT INTO data_sources (
            source_id, source_name, source_type, connection_config, 
            mapping_rules, enabled, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (source_id) DO UPDATE SET
            source_name = EXCLUDED.source_name,
            source_type = EXCLUDED.source_type,
            connection_config = EXCLUDED.connection_config,
            mapping_rules = EXCLUDED.mapping_rules,
            enabled = EXCLUDED.enabled,
            updated_at = EXCLUDED.updated_at
        """
        
        await self.db.execute_command(
            query,
            source_config['source_id'],
            source_config['source_name'],
            source_config['source_type'],
            json.dumps(source_config['connection_config']),
            json.dumps(source_config['mapping_rules']),
            source_config['enabled'],
            source_config['created_at'],
            source_config['updated_at']
        )
        
        print(f"✅ Source '{source_config['source_name']}' configurée avec succès!")
    
    async def list_sources(self):
        """Lister toutes les sources configurées"""
        query = """
        SELECT source_id, source_name, source_type, enabled, created_at
        FROM data_sources
        ORDER BY created_at DESC
        """
        
        sources = await self.db.execute_query(query)
        
        if not sources:
            print("❌ Aucune source de données configurée")
            return
        
        print("\n📊 Sources de données configurées:")
        print("-" * 80)
        print(f"{'ID':<20} {'Nom':<25} {'Type':<15} {'Statut':<10} {'Créée le'}")
        print("-" * 80)
        
        for source in sources:
            status = "✅ Actif" if source['enabled'] else "❌ Inactif"
            created = source['created_at'].strftime('%Y-%m-%d %H:%M')
            print(f"{source['source_id']:<20} {source['source_name']:<25} {source['source_type']:<15} {status:<10} {created}")
    
    async def test_source(self, source_id: str):
        """Tester une source de données"""
        query = "SELECT * FROM data_sources WHERE source_id = $1"
        source = await self.db.execute_query_one(query, source_id)
        
        if not source:
            print(f"❌ Source '{source_id}' non trouvée")
            return
        
        print(f"🧪 Test de la source '{source['source_name']}'...")
        
        try:
            # Importer le connecteur approprié
            source_type = source['source_type']
            if source_type == 'oss_network':
                from workers.connectors.oss_network_connector import OSS_NetworkConnector
                connector_class = OSS_NetworkConnector
            elif source_type == 'itsm_servicenow':
                from workers.connectors.itsm_servicenow_connector import ITSM_ServiceNowConnector
                connector_class = ITSM_ServiceNowConnector
            elif source_type == 'energy_iot':
                from workers.connectors.energy_iot_connector import Energy_IoTConnector
                connector_class = Energy_IoTConnector
            elif source_type == 'snmp':
                from workers.connectors.snmp_connector import SNMPConnector
                connector_class = SNMPConnector
            else:
                print(f"❌ Type de connecteur non supporté: {source_type}")
                return
            
            # Créer une instance du connecteur
            connector = connector_class(
                data_source=source,
                message_producer=None,  # Pas nécessaire pour le test
                quality_validator=None
            )
            
            # Initialiser et tester
            await connector.initialize()
            health = await connector.health_check()
            
            if health['healthy']:
                print(f"✅ Source '{source['source_name']}' est opérationnelle")
                print(f"   Dernière sync: {health.get('last_sync', 'N/A')}")
                print(f"   Nombre d'erreurs: {health.get('error_count', 0)}")
            else:
                print(f"❌ Source '{source['source_name']}' a des problèmes")
                print(f"   Erreur: {health.get('error', 'Inconnue')}")
            
            await connector.disconnect()
            
        except Exception as e:
            print(f"❌ Erreur lors du test: {e}")


async def main():
    """Fonction principale"""
    configurator = DataSourceConfigurator()
    
    if len(sys.argv) < 2:
        print("Usage: python configure_data_sources.py <command> [options]")
        print("\nCommandes disponibles:")
        print("  list                    - Lister les sources configurées")
        print("  test <source_id>        - Tester une source")
        print("  add-oss                 - Ajouter une source OSS (interactif)")
        print("  add-servicenow          - Ajouter une source ServiceNow (interactif)")
        print("  add-iot                 - Ajouter une source IoT (interactif)")
        print("  add-snmp                - Ajouter une source SNMP (interactif)")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        await configurator.list_sources()
    
    elif command == "test":
        if len(sys.argv) < 3:
            print("❌ Usage: python configure_data_sources.py test <source_id>")
            return
        source_id = sys.argv[2]
        await configurator.test_source(source_id)
    
    elif command == "add-oss":
        print("🔧 Configuration d'une source OSS")
        print("=" * 50)
        
        config = {}
        config['source_name'] = input("Nom de la source: ")
        config['base_url'] = input("URL de base (ex: https://oss.company.com): ")
        config['oss_type'] = input("Type OSS (ericsson/huawei/nokia/generic): ")
        
        print("\n🔐 Configuration d'authentification:")
        auth_type = input("Type d'auth (basic/oauth2/session): ")
        config['authentication'] = {'type': auth_type}
        
        if auth_type == 'basic':
            config['authentication']['username'] = input("Nom d'utilisateur: ")
            config['authentication']['password'] = input("Mot de passe: ")
        elif auth_type == 'oauth2':
            config['authentication']['client_id'] = input("Client ID: ")
            config['authentication']['client_secret'] = input("Client Secret: ")
            config['authentication']['token_endpoint'] = input("Token endpoint (optionnel): ")
        elif auth_type == 'session':
            config['authentication']['username'] = input("Nom d'utilisateur: ")
            config['authentication']['password'] = input("Mot de passe: ")
            config['authentication']['login_endpoint'] = input("Login endpoint (optionnel): ")
        
        config['technologies'] = input("Technologies (ex: 4G,5G): ").split(',')
        config['tenant_mapping'] = input("Tenant ID (optionnel): ") or 'default-tenant'
        
        source_id = await configurator.add_oss_source(config)
        print(f"✅ Source OSS configurée avec l'ID: {source_id}")
    
    elif command == "add-servicenow":
        print("🔧 Configuration d'une source ServiceNow")
        print("=" * 50)
        
        config = {}
        config['source_name'] = input("Nom de la source: ")
        config['instance_url'] = input("URL de l'instance (ex: https://company.service-now.com): ")
        
        print("\n🔐 Configuration d'authentification:")
        auth_type = input("Type d'auth (basic/oauth2): ")
        config['authentication'] = {'type': auth_type}
        
        if auth_type == 'basic':
            config['authentication']['username'] = input("Nom d'utilisateur: ")
            config['authentication']['password'] = input("Mot de passe: ")
        elif auth_type == 'oauth2':
            config['authentication']['client_id'] = input("Client ID: ")
            config['authentication']['client_secret'] = input("Client Secret: ")
            config['authentication']['refresh_token'] = input("Refresh Token: ")
        
        config['tables'] = input("Tables à synchroniser (ex: incident,sc_request): ").split(',')
        config['tenant_mapping'] = input("Tenant ID (optionnel): ") or 'default-tenant'
        
        source_id = await configurator.add_servicenow_source(config)
        print(f"✅ Source ServiceNow configurée avec l'ID: {source_id}")
    
    elif command == "add-iot":
        print("🔧 Configuration d'une source IoT")
        print("=" * 50)
        
        config = {}
        config['source_name'] = input("Nom de la source: ")
        config['base_url'] = input("URL de base (ex: https://iot.company.com): ")
        
        print("\n🔐 Configuration d'authentification:")
        auth_type = input("Type d'auth (api_key/basic): ")
        config['authentication'] = {'type': auth_type}
        
        if auth_type == 'api_key':
            config['authentication']['api_key'] = input("Clé API: ")
            config['authentication']['key_header'] = input("Header (optionnel, défaut: X-API-Key): ") or 'X-API-Key'
        elif auth_type == 'basic':
            config['authentication']['username'] = input("Nom d'utilisateur: ")
            config['authentication']['password'] = input("Mot de passe: ")
        
        config['sensor_types'] = input("Types de capteurs (ex: power,voltage,current): ").split(',')
        config['polling_interval'] = int(input("Intervalle de polling en secondes (défaut: 60): ") or 60)
        config['tenant_mapping'] = input("Tenant ID (optionnel): ") or 'default-tenant'
        
        source_id = await configurator.add_iot_source(config)
        print(f"✅ Source IoT configurée avec l'ID: {source_id}")
    
    elif command == "add-snmp":
        print("🔧 Configuration d'une source SNMP")
        print("=" * 50)
        
        config = {}
        config['source_name'] = input("Nom de la source: ")
        
        print("\n🎯 Configuration des cibles SNMP:")
        targets = []
        while True:
            host = input("Host IP (ou 'fin' pour terminer): ")
            if host.lower() == 'fin':
                break
            
            port = int(input("Port (défaut: 161): ") or 161)
            community = input("Community (défaut: public): ") or 'public'
            version = input("Version (1/2c/3, défaut: 2c): ") or '2c'
            
            targets.append({
                'host': host,
                'port': port,
                'community': community,
                'version': version
            })
        
        config['targets'] = targets
        config['oids'] = input("OIDs à surveiller (séparés par virgule): ").split(',')
        config['polling_interval'] = int(input("Intervalle de polling en secondes (défaut: 300): ") or 300)
        config['tenant_mapping'] = input("Tenant ID (optionnel): ") or 'default-tenant'
        
        source_id = await configurator.add_snmp_source(config)
        print(f"✅ Source SNMP configurée avec l'ID: {source_id}")
    
    else:
        print(f"❌ Commande inconnue: {command}")


if __name__ == "__main__":
    asyncio.run(main())
