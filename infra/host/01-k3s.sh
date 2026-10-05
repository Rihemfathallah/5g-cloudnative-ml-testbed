#!/usr/bin/env bash
# Single-node k3s on the host (not k3d): pod network namespaces, Open vSwitch bridges and the CPU
# manager are then all directly reachable from the host (ADR-1).
set -euo pipefail
K3S_VERSION=${K3S_VERSION:-v1.31.4+k3s1}
NCPU=$(nproc)
# Reserve 2 logical CPUs for the OS, k3s, OVS and Ryu; the rest form the pool the static CPU
# manager hands out as exclusive cores to Guaranteed pods (gNB, UE, UPF).
RESERVED=${RESERVED_CPUS:-0-1}
sudo mkdir -p /etc/rancher/k3s
sudo tee /etc/rancher/k3s/config.yaml >/dev/null <<CFG
write-kubeconfig-mode: "0644"
disable: [traefik, servicelb]
kubelet-arg:
  - "cpu-manager-policy=static"
  - "reserved-cpus=${RESERVED}"
  - "kube-reserved=memory=512Mi"
  - "system-reserved=memory=1Gi"
CFG
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sh -
mkdir -p ~/.kube && cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
kubectl wait --for=condition=Ready node --all --timeout=180s
echo "k3s $K3S_VERSION up on $NCPU CPUs (reserved: $RESERVED)"
# The static CPU manager keeps its state in a file; changing the policy later requires deleting
# /var/lib/rancher/k3s/agent/kubelet/cpu_manager_state and restarting k3s.
