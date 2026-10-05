#!/usr/bin/env bash
# Host packages. Ubuntu 22.04/24.04, run once.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y curl git jq make python3-venv python3-pip iperf3 iproute2 \
     mininet openvswitch-switch gettext-base linux-modules-extra-$(uname -r) || \
sudo apt-get install -y curl git jq make python3-venv python3-pip iperf3 iproute2 mininet openvswitch-switch gettext-base
# SCTP for N2 (NGAP). Without it the AMF cannot open its NGAP socket and the gNB never connects.
echo sctp | sudo tee /etc/modules-load.d/sctp.conf
sudo modprobe sctp
# Docker is only used to BUILD images (they are imported into k3s's containerd).
command -v docker >/dev/null || { curl -fsSL https://get.docker.com | sudo sh; sudo usermod -aG docker "$USER"; }
# Helm
command -v helm >/dev/null || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
# Long collection runs: the laptop must not suspend.
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
echo "prereqs OK. Log out/in once if you were just added to the docker group."
