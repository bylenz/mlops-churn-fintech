#!/usr/bin/env bash
# Two-phase AWS deploy: create ECR, push the image, then apply the rest.
# Avoids the empty-registry race where the ECS service can't pull an image.
#
# Usage: ./scripts/deploy_aws.sh '<grafana_admin_password>'
set -euo pipefail

GRAFANA_PW="${1:?Grafana admin password required (min 8 chars)}"
REGION="${AWS_REGION:-us-east-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT/infra"

echo "==> Phase 0: ensure production artifacts exist"
cd "$ROOT"
for f in data/production_model.pkl data/production_model_info.json \
         data/scaler.pkl data/feature_names.csv; do
  test -f "$f" || { echo "MISSING $f — run the local pipeline first"; exit 1; }
done

cd "$ROOT/infra"
terraform init -input=false

echo "==> Phase 1: create ECR only"
terraform apply -auto-approve -input=false \
  -var="grafana_admin_password=${GRAFANA_PW}" \
  -target=aws_ecr_repository.this

ECR_URL=$(terraform output -raw ecr_repository_url)
IMAGE_TAG=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo latest)

echo "==> Phase 2: build + push image ($ECR_URL:$IMAGE_TAG and :latest)"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR_URL"
docker build -t "$ECR_URL:$IMAGE_TAG" -t "$ECR_URL:latest" "$ROOT"
docker push "$ECR_URL:$IMAGE_TAG"
docker push "$ECR_URL:latest"

echo "==> Phase 3: apply full infrastructure"
terraform apply -auto-approve -input=false \
  -var="grafana_admin_password=${GRAFANA_PW}" \
  -var="image_tag=${IMAGE_TAG}"

echo
echo "API URL:     $(terraform output -raw api_url)"
echo "Grafana URL: $(terraform output -raw grafana_url)"
