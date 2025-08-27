REGISTRY ?= local
TAG ?= dev

.PHONY: build all push compose-up k8s-dev k8s-prod helm-backend

build:
	docker build -f backend/Dockerfile -t $(REGISTRY)/sam-backend:$(TAG) .
	docker build -f workers/Dockerfile -t $(REGISTRY)/sam-workers:$(TAG) .
	docker build -f client-portal/Dockerfile -t $(REGISTRY)/sam-client-portal:$(TAG) .
	docker build -f frontend/Dockerfile -t $(REGISTRY)/sam-frontend:$(TAG) .

push:
	docker push $(REGISTRY)/sam-backend:$(TAG) || true
	docker push $(REGISTRY)/sam-workers:$(TAG) || true
	docker push $(REGISTRY)/sam-client-portal:$(TAG) || true
	docker push $(REGISTRY)/sam-frontend:$(TAG) || true

compose-up:
	cd deploy/compose && docker compose -f compose.backend.yml -f compose.workers.yml -f compose.client-portal.yml -f compose.frontend.yml -f compose.observability.yml up -d --build

k8s-dev:
	kubectl apply -k deploy/k8s/overlays/dev

k8s-prod:
	kubectl apply -k deploy/k8s/overlays/prod

helm-backend:
	cd deploy/helm/backend && helm install backend . --namespace sam --create-namespace --set image.repository=$(REGISTRY)/sam-backend --set image.tag=$(TAG)
