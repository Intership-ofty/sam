# 🚀 API KPI - Guide d'utilisation

## 📋 Endpoints disponibles

### **1. Créer un nouveau KPI**
```http
POST /api/v1/kpi/definitions
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  "name": "unique_kpi_id",
  "display_name": "Nom affiché du KPI",
  "description": "Description détaillée",
  "category": "network|energy|operational|financial",
  "unit": "%",
  "formula": "AVG(metric_name)",
  "target_value": 95.0,
  "warning_threshold": 90.0,
  "critical_threshold": 85.0,
  "calculation_interval": "5m",
  "enabled": true,
  "tenant_specific": false,
  "metadata": {
    "source": "custom",
    "priority": "high"
  }
}
```

**Réponse :**
```json
{
  "kpi_id": "uuid",
  "name": "unique_kpi_id",
  "status": "created",
  "message": "KPI definition 'Nom affiché du KPI' created successfully"
}
```

### **2. Lister tous les KPIs**
```http
GET /api/v1/kpi/definitions?category=network&enabled_only=true
Authorization: Bearer YOUR_TOKEN
```

**Paramètres de requête :**
- `category` : Filtrer par catégorie (network, energy, operational, financial)
- `enabled_only` : Afficher seulement les KPIs actifs (défaut: true)

### **3. Obtenir les métriques KPI**
```http
GET /api/v1/kpi/metrics?site_id=site-001&category=network&limit=50
Authorization: Bearer YOUR_TOKEN
```

### **4. Obtenir les tendances KPI**
```http
GET /api/v1/kpi/trends/network_performance?time_range=24h&site_id=site-001
Authorization: Bearer YOUR_TOKEN
```

### **5. Déclencher un calcul manuel**
```http
POST /api/v1/kpi/calculate
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  "site_ids": ["site-001", "site-002"],
  "kpi_names": ["network_performance", "energy_efficiency"],
  "time_range": "24h",
  "priority": "high",
  "requested_by": "user@example.com",
  "recalculate": false
}
```

## 🏗️ Structure d'un KPI

### **Propriétés obligatoires :**
- `name` : Identifiant unique (string)
- `display_name` : Nom affiché (string)
- `description` : Description (string)
- `category` : Catégorie (network|energy|operational|financial)
- `unit` : Unité de mesure (string)
- `formula` : Formule de calcul SQL (string)

### **Propriétés optionnelles :**
- `target_value` : Valeur cible (float)
- `warning_threshold` : Seuil d'avertissement (float)
- `critical_threshold` : Seuil critique (float)
- `calculation_interval` : Intervalle de calcul (1m|5m|15m|30m|1h|6h|1d)
- `enabled` : Actif/Désactivé (boolean, défaut: true)
- `tenant_specific` : Spécifique au tenant (boolean, défaut: false)
- `metadata` : Métadonnées additionnelles (object)

## 📊 Catégories de KPIs

### **🌐 NETWORK (Performance réseau)**
- Métriques de qualité de signal
- Taux de succès des appels
- Débit de données
- Latence réseau
- Disponibilité des cellules

### **⚡ ENERGY (Efficacité énergétique)**
- Consommation d'énergie
- Efficacité des batteries
- Coût énergétique
- Utilisation des énergies renouvelables

### **⚙️ OPERATIONAL (Excellence opérationnelle)**
- Temps de réparation (MTTR)
- Temps entre pannes (MTBF)
- Taux de résolution au premier appel
- Conformité de maintenance

### **💰 FINANCIAL (Performance financière)**
- Revenus par site
- Coûts opérationnels
- ROI des investissements
- Coût par incident

## 🔧 Exemples d'utilisation

### **Exemple 1 : KPI de qualité réseau**
```json
{
  "name": "signal_quality_5g",
  "display_name": "5G Signal Quality",
  "description": "Qualité du signal 5G en dBm",
  "category": "network",
  "unit": "dBm",
  "formula": "AVG(rsrp_5g_dbm)",
  "target_value": -85.0,
  "warning_threshold": -100.0,
  "critical_threshold": -110.0,
  "calculation_interval": "5m",
  "enabled": true,
  "tenant_specific": false,
  "metadata": {
    "technology": "5G",
    "measurement_type": "RSRP"
  }
}
```

### **Exemple 2 : KPI de coût énergétique**
```json
{
  "name": "energy_cost_per_site",
  "display_name": "Energy Cost per Site",
  "description": "Coût énergétique mensuel par site",
  "category": "financial",
  "unit": "$/month",
  "formula": "SUM(energy_cost_monthly) / COUNT(DISTINCT site_id)",
  "target_value": 500.0,
  "warning_threshold": 750.0,
  "critical_threshold": 1000.0,
  "calculation_interval": "1d",
  "enabled": true,
  "tenant_specific": true,
  "metadata": {
    "currency": "USD",
    "billing_period": "monthly"
  }
}
```

### **Exemple 3 : KPI opérationnel**
```json
{
  "name": "maintenance_compliance",
  "display_name": "Maintenance Compliance Rate",
  "description": "Taux de conformité des maintenances préventives",
  "category": "operational",
  "unit": "%",
  "formula": "(completed_pm_tasks / scheduled_pm_tasks) * 100",
  "target_value": 95.0,
  "warning_threshold": 85.0,
  "critical_threshold": 75.0,
  "calculation_interval": "1d",
  "enabled": true,
  "tenant_specific": false,
  "metadata": {
    "task_type": "preventive_maintenance",
    "compliance_window": "30_days"
  }
}
```

## 🚀 Test de l'API

### **1. Utiliser le script de test**
```bash
cd backend
python test_kpi_api.py
```

### **2. Utiliser curl**
```bash
# Créer un KPI
curl -X POST "http://localhost:8000/api/v1/kpi/definitions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "test_kpi",
    "display_name": "Test KPI",
    "description": "KPI de test",
    "category": "network",
    "unit": "%",
    "formula": "AVG(test_metric)",
    "target_value": 95.0
  }'

# Lister les KPIs
curl -X GET "http://localhost:8000/api/v1/kpi/definitions" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### **3. Utiliser Postman**
1. Importer la collection `backend/postman_collection.json`
2. Configurer l'environnement avec votre token
3. Utiliser les requêtes KPI pré-configurées

## ⚠️ Gestion des erreurs

### **Codes de statut courants :**
- **200** : Succès
- **400** : Erreur de validation (KPI existe déjà, paramètres invalides)
- **401** : Non authentifié
- **403** : Accès refusé
- **500** : Erreur serveur

### **Exemple d'erreur :**
```json
{
  "detail": "KPI with name 'existing_kpi' already exists"
}
```

## 🔄 Calcul automatique

Une fois créé, le KPI sera :
1. **Calculé automatiquement** selon l'intervalle défini
2. **Mis en cache** dans Redis pour les performances
3. **Streamé en temps réel** via WebSocket/SSE
4. **Prédit** via ML pour les tendances futures
5. **Alerté** si les seuils sont dépassés

## 📈 Monitoring et alertes

- **Métriques Prometheus** : `kpi_calculations_total`, `kpi_errors_total`
- **Logs structurés** : Tous les calculs sont loggés
- **Alertes automatiques** : Via les seuils définis
- **Dashboards** : Intégration Grafana disponible

---

**L'API KPI est maintenant prête pour créer et gérer des KPIs personnalisés !** 🎉
