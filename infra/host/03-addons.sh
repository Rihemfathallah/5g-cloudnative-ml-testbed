#!/usr/bin/env bash
# Monitoring (kube-prometheus-stack) and Chaos Mesh, pinned chart versions.
set -euo pipefail
cd "$(dirname "$0")/../k8s"
KPS_VERSION=${KPS_VERSION:-66.3.1}
CHAOS_VERSION=${CHAOS_VERSION:-2.7.0}
HOST_IP=$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm repo update
sed "s/__HOST_IP__/${HOST_IP}/" monitoring/values-kube-prometheus-stack.yaml > /tmp/kps-values.yaml
helm upgrade --install kps prometheus-community/kube-prometheus-stack --version "$KPS_VERSION" \
  -n monitoring --create-namespace -f /tmp/kps-values.yaml --wait --timeout 10m
helm upgrade --install chaos-mesh chaos-mesh/chaos-mesh --version "$CHAOS_VERSION" \
  -n chaos-mesh --create-namespace -f chaos/values-chaos-mesh.yaml --wait --timeout 10m
echo "Prometheus: http://localhost:30090   Grafana: http://localhost:30030 (admin / testbed)"
