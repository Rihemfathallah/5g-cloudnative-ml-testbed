#!/usr/bin/env bash
# Secondary networks for N3: Multus (thick) + OVS-CNI + the reference `static` IPAM plugin.
# k3s keeps CNI binaries and configs in non-standard paths; every manifest below is patched for them.
set -euo pipefail
MULTUS_VERSION=${MULTUS_VERSION:-v4.1.4}
OVSCNI_VERSION=${OVSCNI_VERSION:-v0.38.2}
CNI_PLUGINS_VERSION=${CNI_PLUGINS_VERSION:-v1.6.2}
K3S_CNI_BIN=/var/lib/rancher/k3s/data/cni
K3S_CNI_CONF=/var/lib/rancher/k3s/agent/etc/cni/net.d
sudo mkdir -p "$K3S_CNI_BIN"
# 1) reference plugins (we need `static`)
curl -fsSL "https://github.com/containernetworking/plugins/releases/download/${CNI_PLUGINS_VERSION}/cni-plugins-linux-amd64-${CNI_PLUGINS_VERSION}.tgz" \
  | sudo tar -xz -C "$K3S_CNI_BIN" ./static
# 2) Multus thick, paths rewritten for k3s
curl -fsSL "https://raw.githubusercontent.com/k8snetworkplumbingwg/multus-cni/${MULTUS_VERSION}/deployments/multus-daemonset-thick.yml" \
  | sed -e "s|/etc/cni/net.d|${K3S_CNI_CONF}|g" -e "s|/opt/cni/bin|${K3S_CNI_BIN}|g" \
  | kubectl apply -f -
# 3) OVS-CNI plugin + marker
curl -fsSL "https://github.com/k8snetworkplumbingwg/ovs-cni/releases/download/${OVSCNI_VERSION}/ovs-cni.yml" \
  | sed -e "s|/opt/cni/bin|${K3S_CNI_BIN}|g" | kubectl apply -f -
kubectl -n kube-system rollout status ds/kube-multus-ds --timeout=180s
echo "check: ls $K3S_CNI_BIN | grep -E 'multus|ovs|static'"
ls "$K3S_CNI_BIN" | grep -E 'multus|ovs|static' || true
