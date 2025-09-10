# Port Mapping for Workers

## 🎯 **PORTS ASSIGNÉS**

| Worker | Port | Container |
|--------|------|-----------|
| **data-ingestor** | 8001 | sam-data-ingestor |
| **kpi-worker** | 8002 | sam-kpi-worker |
| **anomaly-detector** | 8003 | sam-anomaly-detector |
| **bi-engine** | 8004 | sam-bi-engine |
| **event-correlator** | 8005 | sam-event-correlator |
| **rca-analyzer** | 8006 | sam-rca-analyzer |
| **noc-orchestrator** | 8007 | sam-noc-orchestrator |
| **optimization-engine** | 8008 | sam-optimization-engine |
| **predictive-maintenance** | 8009 | sam-predictive-maintenance |

## 🚨 **CONFLITS DÉTECTÉS**

### ❌ **Avant Correction:**
- **Port 8004**: bi-engine ET event-correlator (CONFLIT)
- **Port 8005**: event-correlator ET rca-analyzer (CONFLIT)

### ✅ **Après Correction:**
- **Port 8004**: bi-engine UNIQUEMENT
- **Port 8005**: event-correlator UNIQUEMENT  
- **Port 8006**: rca-analyzer UNIQUEMENT

## 🔧 **HEALTHCHECK NATIF**
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:XXXX/health')"]
```

**Avantages:**
- ✅ **Pas de curl** requis
- ✅ **Python natif** dans l'image
- ✅ **urllib.request** standard
- ✅ **Léger** et fiable
