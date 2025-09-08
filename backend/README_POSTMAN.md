# 🚀 Guide Postman - Towerco AIOps Backend API

## 📋 Installation

1. **Importer la collection** :
   - Ouvrir Postman
   - Cliquer sur "Import"
   - Sélectionner le fichier `postman_collection.json`

2. **Configurer les variables** :
   - `base_url` : `http://localhost:8000` (déjà configuré)
   - `auth_token` : Token d'authentification (à remplir après login)

## 🔐 Authentification

### 1. Obtenir la configuration d'authentification
```
GET {{base_url}}/api/v1/auth/config
```

### 2. Se connecter (obtenir l'URL de login)
```
GET {{base_url}}/api/v1/auth/login
```
**Réponse** : `{"login_url": "http://localhost:8081/realms/towerco/protocol/openid-connect/auth?..."}`

### 3. Récupérer le token (après redirection Keycloak)
```
POST {{base_url}}/api/v1/auth/callback
Body: {
  "code": "code_from_keycloak",
  "state": "optional_state"
}
```
**Réponse** : `{"access_token": "...", "user": {...}}`

### 4. Utiliser le token
- Copier le `access_token` de la réponse
- Coller dans la variable `{{auth_token}}`
- Toutes les requêtes suivantes utiliseront automatiquement ce token

## 📊 Tests par Catégorie

### Health Check
- **GET /health** - Vérifier que l'API est accessible

### KPIs (Indicateurs de Performance)
- **GET /api/v1/kpi/metrics** - Récupérer les métriques KPI
- **GET /api/v1/kpi/trends/{kpi_name}** - Obtenir les tendances d'un KPI
- **GET /api/v1/kpi/alerts** - Lister les alertes KPI
- **GET /api/v1/kpi/stream/{kpi_name}** - Stream temps réel (SSE)

### Sites (Sites de télécommunication)
- **GET /api/v1/sites** - Lister tous les sites
- **GET /api/v1/sites/{site_id}** - Détails d'un site
- **GET /api/v1/sites/{site_id}/health** - Santé d'un site
- **GET /api/v1/sites/{site_id}/kpis/latest** - KPIs récents d'un site
- **GET /api/v1/sites/{site_id}/alerts/active** - Alertes actives d'un site

### Administration
- **GET /api/v1/admin/status** - Statut du système
- **GET /api/v1/admin/metrics** - Métriques système

### Événements
- **GET /api/v1/events** - Événements récents

### Rapports
- **POST /api/v1/reports/generate** - Générer un rapport
- **GET /api/v1/reports/status/{report_id}** - Statut d'un rapport

### Notifications
- **GET /api/v1/notifications** - Notifications utilisateur
- **PUT /api/v1/notifications/mark-read** - Marquer comme lu

## 🔧 Paramètres de Requête Courants

### Pagination
- `limit` : Nombre d'éléments (défaut: 50)
- `offset` : Décalage pour la pagination

### Filtres KPI
- `category` : network, energy, operational, financial
- `site_id` : ID du site spécifique
- `time_range` : 1h, 24h, 7d, 30d

### Filtres Alertes
- `severity` : critical, major, minor, warning
- `status` : active, acknowledged, resolved

### Filtres Sites
- `region` : Europe, Asia, Americas
- `technology` : 5G, 4G, LTE
- `status` : active, inactive, maintenance

## 📝 Exemples de Réponses

### KPI Metrics
```json
{
  "data": [
    {
      "kpi_name": "Network Performance",
      "current_value": 95.2,
      "target_value": 90.0,
      "unit": "%",
      "category": "network",
      "trend": "up",
      "quality_score": 0.95,
      "last_calculated": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Site Health
```json
{
  "data": {
    "site_id": "site-001",
    "health_score": 0.92,
    "status": "excellent",
    "active_alerts": 2,
    "recent_metrics": 15,
    "categories": {
      "network": {"kpi_count": 5, "avg_quality": 0.94},
      "energy": {"kpi_count": 3, "avg_quality": 0.89}
    }
  }
}
```

## ⚠️ Gestion des Erreurs

### Codes de Statut Courants
- **200** : Succès
- **401** : Non authentifié (token manquant/invalide)
- **403** : Accès refusé (permissions insuffisantes)
- **404** : Ressource non trouvée
- **422** : Erreur de validation
- **500** : Erreur serveur

### Format d'Erreur
```json
{
  "detail": "Description de l'erreur",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🚀 Tests de Performance

### Test de Charge Simple
1. Créer un environnement avec `base_url` et `auth_token`
2. Utiliser la collection Runner de Postman
3. Configurer des itérations pour tester la charge

### Test de Stream SSE
1. Utiliser l'endpoint `/api/v1/kpi/stream/{kpi_name}`
2. Observer les données en temps réel
3. Tester la reconnexion automatique

## 📚 Ressources Supplémentaires

- **Documentation API** : `http://localhost:8000/docs` (Swagger UI)
- **ReDoc** : `http://localhost:8000/redoc`
- **Métriques Prometheus** : `http://localhost:8000/metrics`

## 🔍 Debugging

### Logs Backend
```bash
docker logs compose-backend-1 -f
```

### Vérifier la Base de Données
```bash
docker exec -it compose-postgres-1 psql -U towerco -d towerco_aiops
```

### Vérifier Keycloak
```bash
docker logs keycloak -f
```

---

**Note** : Assurez-vous que tous les services Docker sont démarrés avant de tester les APIs.
