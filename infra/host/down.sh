#!/usr/bin/env bash
# Reverse order: pods first, then (manually) the SDN ring. Stopping Mininet while the pods run
# deletes the OVS bridges under their N3 interfaces.
set -euo pipefail
kubectl delete stresschaos,podchaos,networkchaos -n 5g --all --ignore-not-found
kubectl delete namespace 5g --ignore-not-found --wait=true
echo "pods gone; you can now stop the SDN ring (Ctrl+C in mininet_service.py, then ryu-manager)"
