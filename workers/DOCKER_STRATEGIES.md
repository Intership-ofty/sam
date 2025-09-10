# Docker Strategies pour Workers

## 🎯 **PROBLÈME RÉSOLU**

### **❌ AVANT: Images Séparées**
- **9 Dockerfiles** différents
- **9 Images** de ~1GB chacune
- **Taille totale**: ~9GB
- **Duplication** massive des packages ML/IA

### **✅ APRÈS: Image Unifiée**
- **1 Dockerfile.unified** 
- **1 Image** de ~1GB réutilisée
- **Taille totale**: ~1GB
- **Économie**: **-8GB (88%)**

## 📁 **DOCKERFILES DISPONIBLES**

### **1. Image Unifiée (RECOMMANDÉE)**
```yaml
# deploy/compose/compose.workers.yml
dockerfile: workers/Dockerfile.unified
image: sam-workers-unified:${TAG:-dev}
```
- ✅ **1 seule image** pour tous les workers
- ✅ **Tous les packages ML** inclus
- ✅ **Backend core** inclus
- ✅ **Optimisé** pour déploiement

### **2. Images Spécialisées (OPTION)**
```yaml
# Pour use cases spécifiques
data-ingestor:   dockerfile: workers/Dockerfile.ingestor
kpi-worker:      dockerfile: workers/Dockerfile.kpi  
anomaly-detector: dockerfile: workers/Dockerfile.aiops
# Autres workers: dockerfile: workers/Dockerfile.unified
```

## 🔧 **CONFIGURATION ACTUELLE**

### **✅ CORRECTIONS APPLIQUÉES**
1. **Backend Imports**: `COPY backend/ ./backend/` ajouté à tous les Dockerfiles
2. **Python Path**: `ENV PYTHONPATH=/app` configuré
3. **Commands**: `CMD ["python", "workers/worker_name.py"]` corrigé
4. **Image Unifiée**: `compose.workers.yml` utilise `Dockerfile.unified`

### **📦 PACKAGES OPTIMISÉS**
- **Airflow 3.0.4** (base commune)
- **ML Stack**: sklearn, xgboost, pandas, numpy
- **Database**: asyncpg, sqlalchemy
- **Kafka**: aiokafka
- **Monitoring**: prometheus_client

## 🚀 **UTILISATION**

### **Déploiement Optimal (Image Unifiée)**
```bash
docker-compose -f deploy/compose/compose.workers.yml up -d
```

### **Déploiement Spécialisé (si nécessaire)**
```yaml
# Modifier compose.workers.yml pour worker spécifique:
kpi-worker:
  dockerfile: workers/Dockerfile.kpi  # Au lieu de .unified
```

## 📊 **MÉTRIQUES**

| Stratégie | Images | Taille/Image | Taille Totale | Build Time |
|-----------|--------|--------------|---------------|------------|
| **Séparées** | 9 | ~1GB | ~9GB | ~45min |
| **Unifiée** | 1 | ~1GB | ~1GB | ~15min |
| **Économie** | -8 | 0GB | **-8GB** | **-30min** |

## 🎖️ **RECOMMANDATION**

**Utilisez `Dockerfile.unified`** pour:
- ✅ **Production** (économie ressources)
- ✅ **Développement** (build plus rapide)
- ✅ **CI/CD** (moins de transferts)

**Utilisez Dockerfiles spécialisés** uniquement si:
- 🔧 **Contraintes spécifiques** par worker
- 🔧 **Optimisation fine** requise
- 🔧 **Déploiement sélectif** de workers
