# Towerco AIOps Platform - Frontend Applications

## 🚀 **Vue d'ensemble**

Le projet Towerco AIOps dispose maintenant de **deux interfaces frontend professionnelles** :

1. **Client Portal** (React/TypeScript) - Interface moderne pour les utilisateurs avancés
2. **Frontend HTML** - Interface simple et rapide pour la démonstration

## 📋 **Prérequis**

- Docker et Docker Compose
- Node.js 18+ (pour le Client Portal)
- Python 3.8+ (pour le serveur HTTP)

## 🚀 **Démarrage rapide**

### **Option 1 : Script automatique (Recommandé)**

**Windows :**
```bash
start-frontend.bat
```

**Linux/Mac :**
```bash
./start-frontend.sh
```

### **Option 2 : Démarrage manuel**

1. **Démarrer le backend :**
```bash
cd deploy/compose
docker compose -f compose.backend.yml up -d
```

2. **Démarrer le Frontend HTML :**
```bash
cd frontend
python -m http.server 8090
```

3. **Démarrer le Client Portal :**
```bash
cd client-portal
npm install
npm run dev
```

## 🌐 **Accès aux interfaces**

- **Frontend HTML** : http://localhost:8090
- **Client Portal** : http://localhost:5173
- **Backend API** : http://localhost:8000
- **Keycloak** : http://localhost:8080

## 🔐 **Authentification**

### **Keycloak (Production)**
- Interface de login professionnelle
- Gestion des utilisateurs et rôles
- Sécurité enterprise-grade

### **Mode Développement**
- Authentification simple pour les tests
- Pas de configuration Keycloak requise

## 📱 **Fonctionnalités**

### **Client Portal (React/TypeScript)**
- ✅ Dashboard temps réel avec métriques
- ✅ Gestion complète des KPIs
- ✅ Interface de gestion des sites
- ✅ Système d'alertes avancé
- ✅ Design responsive et moderne
- ✅ Authentification Keycloak intégrée

### **Frontend HTML**
- ✅ Page d'accueil professionnelle
- ✅ Dashboard KPI temps réel
- ✅ Interface de gestion des sites
- ✅ Navigation intuitive
- ✅ Design moderne et responsive
- ✅ Authentification simple

## 🔧 **Configuration**

### **Variables d'environnement**

Le fichier `deploy/compose/common.env` contient :
```env
# CORS configuration for frontends
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8090,http://localhost:8000

# Authentication mode
AUTH_MODE=keycloak  # Options: simple, keycloak, none
```

### **APIs disponibles**

- **KPIs** : `/api/v1/kpi/metrics`, `/api/v1/kpi/trends/{kpi_name}`, `/api/v1/kpi/alerts`
- **Sites** : `/api/v1/sites/`, `/api/v1/sites/{site_id}`, `/api/v1/sites/{site_id}/health`
- **Auth** : `/api/v1/auth/login`, `/api/v1/auth/me`, `/api/v1/auth/config`

## 🐛 **Dépannage**

### **Problèmes courants**

1. **Backend non accessible**
   - Vérifiez que Docker est démarré
   - Vérifiez les logs : `docker compose logs backend`

2. **Erreurs CORS**
   - Vérifiez `ALLOWED_ORIGINS` dans `common.env`
   - Redémarrez le backend après modification

3. **Authentification échoue**
   - Vérifiez que Keycloak est démarré
   - Utilisez le mode développement pour les tests

4. **Données non chargées**
   - Vérifiez la connexion à la base de données
   - Vérifiez les logs du backend

### **Logs utiles**

```bash
# Backend logs
docker compose -f deploy/compose/compose.backend.yml logs -f backend

# Keycloak logs
docker compose -f deploy/compose/compose.backend.yml logs -f keycloak

# Base de données logs
docker compose -f deploy/compose/compose.backend.yml logs -f postgres
```

## 📊 **Données de test**

### **Mode Développement**
- Utilise des données mockées pour la démonstration
- Pas de base de données requise

### **Mode Production**
- Connecté aux vraies APIs backend
- Données réelles de la base de données
- Authentification Keycloak

## 🔄 **Mise à jour**

Pour mettre à jour les frontends :

1. **Client Portal :**
```bash
cd client-portal
npm update
npm run build
```

2. **Frontend HTML :**
- Les fichiers sont directement modifiables
- Pas de build requis

## 📈 **Prochaines étapes**

1. **Tests** : Ajouter des tests unitaires et d'intégration
2. **CI/CD** : Automatiser le déploiement
3. **Monitoring** : Ajouter des métriques frontend
4. **PWA** : Transformer en Progressive Web App
5. **Mobile** : Application mobile native

## 🆘 **Support**

Pour toute question ou problème :
- Vérifiez les logs du backend
- Consultez la documentation API : http://localhost:8000/docs
- Vérifiez la configuration dans `deploy/compose/common.env`

---

**Les deux frontends sont maintenant des interfaces professionnelles complètes et prêtes pour la production !** 🎉
