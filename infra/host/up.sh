#!/usr/bin/env bash
# Bring the testbed up in dependency order. Usage: infra/host/up.sh [k8s-srsran|k8s-ueransim|k8s-oai]
#   0. the SDN ring must already run:  sudo python3 infra/sdn/mininet_service.py   (+ ryu-manager)
set -euo pipefail
PROFILE=${1:-k8s-srsran}
cd "$(dirname "$0")/../k8s"
KZ="kubectl kustomize --load-restrictor LoadRestrictionsNone"
for br in s1 s3; do sudo ovs-vsctl br-exists "$br" || { echo "OVS bridge $br missing: start the SDN ring first"; exit 1; }; done
python3 ../open5gs/render.py k8s >/dev/null
$KZ base | kubectl apply -f -
$KZ nad | kubectl apply -f -
$KZ open5gs | kubectl apply -f -
kubectl -n 5g wait --for=condition=complete job/provision-subscribers --timeout=300s
kubectl -n 5g wait --for=condition=Available deploy --all --timeout=300s
case "$PROFILE" in
  k8s-srsran)   $KZ ran-srsran | kubectl apply -f - ;;
  k8s-ueransim) $KZ ran-ueransim | kubectl apply -f - ;;
  k8s-oai)      OAI_TAG=$(grep -oP 'OAI_TAG=\K\S+' ../VERSIONS.md) $KZ ran-oai | envsubst '${OAI_TAG}' | kubectl apply -f - ;;
esac
kubectl apply -f monitoring/podmonitors.yaml
kubectl -n 5g wait --for=condition=Available deploy --all --timeout=300s
# New OVS ports (the pods' n3 interfaces) pass through STP listening + learning (2 x 15 s)
# before they forward. Preflight would otherwise fail on a freshly started testbed.
echo "waiting 45 s for STP to put the new N3 ports into FORWARD"; sleep 45
kubectl -n 5g get pods -o wide
