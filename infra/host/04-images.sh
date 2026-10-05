#!/usr/bin/env bash
# Build every image and import it into k3s's containerd (pods use imagePullPolicy: Never, so a
# pod can never silently run an image other than the one built from this repository).
set -euo pipefail
cd "$(dirname "$0")/../docker"
build() { docker build -t "$1" "$2"; docker save "$1" | sudo k3s ctr images import -; }
build testbed/open5gs:2.7.7       open5gs
build testbed/probe:1.0           probe
build testbed/ueransim:3.3.0      ueransim
build testbed/srsran-gnb:25.10    srsran-gnb     # ~20-40 min on a laptop
build testbed/srsue:23.11         srsue
sudo k3s ctr images ls | grep testbed/
