#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration du backend
"""

import os
import sys

# Ajouter le chemin du backend
sys.path.append('backend')

def test_config():
    """Tester la configuration du backend"""
    try:
        print("🧪 Test de la configuration du backend...")
        
        # Définir les variables d'environnement pour le test
        os.environ['DATABASE_URL'] = 'postgresql+asyncpg://sam:sam@localhost:5432/sam'
        os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
        os.environ['ALLOWED_ORIGINS'] = 'http://localhost:3000,http://localhost:5173,http://localhost:8090,http://localhost:8000'
        
        # Importer et tester la configuration
        from core.config import settings
        
        print("✅ Configuration chargée avec succès!")
        print(f"   ALLOWED_ORIGINS: {settings.ALLOWED_ORIGINS}")
        print(f"   DATABASE_URL: {settings.DATABASE_URL}")
        print(f"   REDIS_URL: {settings.REDIS_URL}")
        print(f"   AUTH_MODE: {settings.AUTH_MODE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
