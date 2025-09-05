# Guide de Connexion aux Sources de Données - Towerco AIOps

## 🔌 **Vue d'ensemble de l'architecture de connexion**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sources de    │    │   Connecteurs   │    │   AIOps         │
│   Données       │    │   Towerco       │    │   Platform      │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │    OSS      │ │───▶│ │   OSS       │ │───▶│ │   Workers   │ │
│ │  (Network)  │ │    │ │ Connector   │ │    │ │  (AI/ML)    │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │    ITSM     │ │───▶│ │   ITSM      │ │───▶│ │   APIs      │ │
│ │ (ServiceNow)│ │    │ │ Connector   │ │    │ │ (Backend)   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │     IoT     │ │───▶│ │    IoT      │ │───▶│ │  Frontends  │ │
│ │ (Sensors)   │ │    │ │ Connector   │ │    │ │   (UI/UX)   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ **Connecteurs disponibles**

### **1. OSS Network Connector**
**Fichier** : `workers/connectors/oss_network_connector.py`

**Sources supportées** :
- Ericsson OSS-RC
- Huawei U2000
- Nokia NetAct
- ZTE ZXONE
- Systèmes OSS génériques

**Types de données** :
- Métriques de performance réseau
- Alarmes et événements
- Configuration des équipements
- Topologie réseau

### **2. ITSM ServiceNow Connector**
**Fichier** : `workers/connectors/itsm_servicenow_connector.py`

**Sources supportées** :
- ServiceNow (instances cloud et on-premise)
- Intégration bidirectionnelle

**Types de données** :
- Incidents
- Demandes de service
- Demandes de changement
- Problèmes
- Base de connaissances

### **3. IoT Energy Connector**
**Fichier** : `workers/connectors/energy_iot_connector.py`

**Sources supportées** :
- Capteurs IoT d'énergie
- Systèmes de gestion d'énergie
- Compteurs intelligents

### **4. SNMP Connector**
**Fichier** : `workers/connectors/snmp_connector.py`

**Sources supportées** :
- Équipements réseau SNMP
- MIBs standard et propriétaires

### **5. REST Connector**
**Fichier** : `workers/connectors/rest_connector.py`

**Sources supportées** :
- APIs REST génériques
- Webhooks
- Intégrations personnalisées

## 🚀 **Comment configurer les connexions**

### **Étape 1 : Configuration de la base de données**

D'abord, ajoutez les sources de données dans la base de données :

```sql
-- Exemple de configuration OSS Ericsson
INSERT INTO data_sources (
    source_id, source_name, source_type, connection_config, mapping_rules, 
    enabled, created_at, updated_at
) VALUES (
    'oss-ericsson-001',
    'Ericsson OSS-RC Production',
    'oss_network',
    '{
        "base_url": "https://oss-ericsson.company.com",
        "oss_type": "ericsson",
        "api_version": "v1",
        "authentication": {
            "type": "oauth2",
            "client_id": "your_client_id",
            "client_secret": "your_client_secret",
            "token_endpoint": "/auth/oauth2/token"
        },
        "polling_endpoints": [
            "/pm/measurements",
            "/fm/alarms"
        ],
        "technologies": ["4G", "5G"],
        "measurement_types": ["throughput", "latency", "availability"],
        "site_filter": "region=North",
        "max_alarms": 1000
    }',
    '{
        "tenant_mapping": "tenant-001",
        "site_mapping": {
            "field": "siteId",
            "mapping": {
                "SITE001": "site-001",
                "SITE002": "site-002"
            }
        }
    }',
    true,
    NOW(),
    NOW()
);

-- Exemple de configuration ServiceNow
INSERT INTO data_sources (
    source_id, source_name, source_type, connection_config, mapping_rules,
    enabled, created_at, updated_at
) VALUES (
    'servicenow-prod-001',
    'ServiceNow Production',
    'itsm_servicenow',
    '{
        "instance_url": "https://company.service-now.com",
        "api_version": "v1",
        "authentication": {
            "type": "basic",
            "username": "aiops_user",
            "password": "secure_password"
        },
        "tables": ["incident", "sc_request", "change_request"],
        "sync_mode": "incremental",
        "max_records": 1000,
        "table_filters": {
            "incident": "state!=7^priority<=3",
            "sc_request": "state!=3^state!=4^state!=5"
        }
    }',
    '{
        "tenant_mapping": "tenant-001",
        "site_field_mapping": "cmdb_ci.name"
    }',
    true,
    NOW(),
    NOW()
);
```

### **Étape 2 : Configuration des workers**

Créez un fichier de configuration pour les workers :

```yaml
# workers/config/data_sources.yaml
data_sources:
  - source_id: "oss-ericsson-001"
    connector_class: "OSS_NetworkConnector"
    polling_interval: 300  # 5 minutes
    enabled: true
    
  - source_id: "servicenow-prod-001"
    connector_class: "ITSM_ServiceNowConnector"
    polling_interval: 600  # 10 minutes
    enabled: true
    
  - source_id: "iot-energy-001"
    connector_class: "Energy_IoTConnector"
    polling_interval: 60   # 1 minute
    enabled: true
```

### **Étape 3 : Démarrage des workers**

```bash
# Démarrer le worker de données
cd workers
python data_ingestor.py --config config/data_sources.yaml

# Ou via Docker
docker run -d --name towerco-data-ingestor \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  towerco-aiops:latest \
  python data_ingestor.py --config config/data_sources.yaml
```

## 🔧 **Configuration détaillée par type de source**

### **1. OSS Network Systems**

#### **Ericsson OSS-RC**
```json
{
  "base_url": "https://oss-ericsson.company.com",
  "oss_type": "ericsson",
  "authentication": {
    "type": "oauth2",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "token_endpoint": "/auth/oauth2/token"
  },
  "polling_endpoints": [
    "/pm/measurements",
    "/fm/alarms"
  ],
  "technologies": ["4G", "5G"],
  "measurement_types": ["throughput", "latency", "availability"],
  "site_filter": "region=North"
}
```

#### **Huawei U2000**
```json
{
  "base_url": "https://u2000-huawei.company.com",
  "oss_type": "huawei",
  "authentication": {
    "type": "session",
    "username": "aiops_user",
    "password": "secure_password",
    "login_endpoint": "/rest/login"
  },
  "polling_endpoints": [
    "/restconf/data/performance",
    "/restconf/data/alarms"
  ],
  "technologies": ["4G", "5G"]
}
```

#### **Nokia NetAct**
```json
{
  "base_url": "https://netact-nokia.company.com",
  "oss_type": "nokia",
  "authentication": {
    "type": "basic",
    "username": "aiops_user",
    "password": "secure_password"
  },
  "polling_endpoints": [
    "/rest/api/v1/measurements",
    "/rest/api/v1/alarms"
  ],
  "technologies": ["4G", "5G"]
}
```

### **2. ITSM ServiceNow**

#### **Configuration de base**
```json
{
  "instance_url": "https://company.service-now.com",
  "api_version": "v1",
  "authentication": {
    "type": "basic",
    "username": "aiops_user",
    "password": "secure_password"
  },
  "tables": ["incident", "sc_request", "change_request"],
  "sync_mode": "incremental",
  "max_records": 1000
}
```

#### **Configuration OAuth2**
```json
{
  "instance_url": "https://company.service-now.com",
  "api_version": "v1",
  "authentication": {
    "type": "oauth2",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "refresh_token": "your_refresh_token"
  },
  "tables": ["incident", "sc_request", "change_request"],
  "sync_mode": "incremental"
}
```

### **3. IoT Energy Systems**

```json
{
  "base_url": "https://iot-energy.company.com",
  "authentication": {
    "type": "api_key",
    "api_key": "your_api_key",
    "key_header": "X-API-Key"
  },
  "sensor_types": ["power", "voltage", "current", "temperature"],
  "polling_interval": 60,
  "batch_size": 100
}
```

### **4. SNMP Devices**

```json
{
  "targets": [
    {
      "host": "192.168.1.100",
      "port": 161,
      "community": "public",
      "version": "2c"
    }
  ],
  "oids": [
    "1.3.6.1.2.1.1.1.0",  # sysDescr
    "1.3.6.1.2.1.1.3.0",  # sysUpTime
    "1.3.6.1.2.1.2.2.1.10" # ifInOctets
  ],
  "polling_interval": 300
}
```

## 📊 **Types de données collectées**

### **1. Métriques réseau**
```json
{
  "data_type": "network_metric",
  "timestamp": "2024-01-15T10:30:00Z",
  "technology": "5G",
  "metric_name": "throughput",
  "metric_value": 150.5,
  "unit": "Mbps",
  "site_id": "site-001",
  "quality_score": 0.95,
  "metadata": {
    "endpoint": "/pm/measurements",
    "oss_type": "ericsson",
    "cell_id": "CELL-001",
    "slice_id": "eMBB"
  }
}
```

### **2. Événements/Alertes**
```json
{
  "data_type": "event",
  "timestamp": "2024-01-15T10:30:00Z",
  "event_type": "ALARM",
  "severity": "CRITICAL",
  "title": "Cell Down",
  "description": "Cell site SITE-001 is down",
  "site_id": "site-001",
  "source_system": "OSS-ericsson",
  "acknowledged": false,
  "metadata": {
    "alarm_id": "ALM-001",
    "probable_cause": "Equipment Failure",
    "managed_object": "CELL-001"
  }
}
```

### **3. Tickets ITSM**
```json
{
  "data_type": "event",
  "event_type": "INCIDENT",
  "title": "Network Outage - Site 001",
  "description": "Complete network outage at site 001",
  "severity": "CRITICAL",
  "status": "In Progress",
  "assigned_to": "John Doe",
  "opened_at": "2024-01-15T09:00:00Z",
  "site_code": "SITE-001",
  "metadata": {
    "servicenow_table": "incident",
    "servicenow_sys_id": "abc123def456",
    "category": "Network",
    "assignment_group": "Network Operations"
  }
}
```

## 🔄 **Flux de données**

### **1. Collecte**
```
Source de données → Connecteur → Validation → Enrichissement → Stockage
```

### **2. Traitement**
```
Données brutes → Workers AI/ML → KPIs → Alertes → APIs
```

### **3. Visualisation**
```
APIs → Frontends → Utilisateurs
```

## 🛡️ **Sécurité et authentification**

### **1. Authentification OAuth2**
```python
# Configuration OAuth2 pour Ericsson OSS-RC
auth_config = {
    "type": "oauth2",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "token_endpoint": "/auth/oauth2/token",
    "scope": "read write"
}
```

### **2. Authentification par clé API**
```python
# Configuration API Key pour IoT
auth_config = {
    "type": "api_key",
    "api_key": "your_api_key",
    "key_header": "X-API-Key"
}
```

### **3. Authentification de base**
```python
# Configuration Basic Auth pour Nokia NetAct
auth_config = {
    "type": "basic",
    "username": "aiops_user",
    "password": "secure_password"
}
```

## 📈 **Monitoring et surveillance**

### **1. Health Checks**
```python
# Vérifier la santé des connecteurs
health = await connector.health_check()
print(f"Connector healthy: {health['healthy']}")
print(f"Last sync: {health['last_sync']}")
print(f"Error count: {health['error_count']}")
```

### **2. Métriques de performance**
- Nombre de records collectés
- Temps de réponse des sources
- Taux d'erreur
- Qualité des données

### **3. Alertes de connectivité**
- Connexion perdue
- Authentification échouée
- Données corrompues
- Dépassement de quota

## 🚀 **Déploiement en production**

### **1. Configuration Docker Compose**
```yaml
# deploy/compose/compose.workers.yml
version: '3.8'
services:
  data-ingestor:
    build:
      context: ../..
      dockerfile: workers/Dockerfile.ingestor
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/sam
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BROKERS=kafka:9092
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
      - kafka
    restart: unless-stopped
```

### **2. Configuration Kubernetes**
```yaml
# deploy/k8s/workers/data-ingestor.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-ingestor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: data-ingestor
  template:
    metadata:
      labels:
        app: data-ingestor
    spec:
      containers:
      - name: data-ingestor
        image: towerco-aiops:latest
        command: ["python", "data_ingestor.py"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aiops-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 🔧 **Dépannage**

### **1. Problèmes de connexion**
```bash
# Vérifier les logs des connecteurs
docker logs towerco-data-ingestor

# Tester la connexion manuellement
python -c "
from workers.connectors.oss_network_connector import OSS_NetworkConnector
connector = OSS_NetworkConnector(config)
await connector.test_connection()
"
```

### **2. Problèmes d'authentification**
- Vérifier les credentials
- Vérifier les URLs des endpoints
- Vérifier les permissions utilisateur

### **3. Problèmes de données**
- Vérifier le format des données
- Vérifier les mappings de champs
- Vérifier la qualité des données

## 📚 **Exemples d'utilisation**

### **1. Ajouter une nouvelle source OSS**
```python
# Ajouter une source OSS personnalisée
source_config = {
    "source_id": "oss-custom-001",
    "source_name": "Custom OSS System",
    "source_type": "oss_network",
    "connection_config": {
        "base_url": "https://custom-oss.company.com",
        "oss_type": "generic",
        "authentication": {
            "type": "basic",
            "username": "user",
            "password": "pass"
        },
        "polling_endpoints": ["/api/metrics", "/api/alarms"]
    },
    "mapping_rules": {
        "tenant_mapping": "tenant-001",
        "site_mapping": {"field": "siteId"}
    }
}
```

### **2. Créer un connecteur personnalisé**
```python
# workers/connectors/custom_connector.py
from .base_connector import DataConnector, HttpConnectorMixin

class CustomConnector(DataConnector, HttpConnectorMixin):
    async def validate_config(self):
        # Validation spécifique
        pass
    
    async def connect(self):
        # Connexion spécifique
        pass
    
    async def fetch_data(self):
        # Collecte de données spécifique
        pass
```

---

**Avec cette architecture, l'AIOps Towerco peut se connecter à pratiquement n'importe quelle source de données telecom et ITSM !** 🚀
