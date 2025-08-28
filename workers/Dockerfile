FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     gcc g++ libpq-dev curl     && rm -rf /var/lib/apt/lists/*

# Copy requirements (reuse backend requirements if referenced)
# COPY backend/requirements.txt backend/requirements.txt
COPY workers/requirements.txt workers/requirements.txt
RUN pip install --no-cache-dir -r workers/requirements.txt

COPY workers/ ./workers/
# COPY backend ./backend

# Choose the worker at runtime with: WORKER=ingestor|kpi|anomaly|noc|predictive|optimizer|correlator|bi
ENV WORKER=ingestor
CMD ["bash","-lc","python -m workers.$WORKER" ]
