# Pinned versions

Every component is pinned. Change a version only in its own commit, and note it in LOG.md: a
version bump invalidates comparisons with earlier runs.

| Component | Version | Where it is used | Validated in this repo |
|---|---|---|---|
| Ubuntu (host) | 22.04 or 24.04 LTS | host | -- |
| k3s | v1.31.x+k3s1 (`K3S_VERSION` in host/01-k3s.sh) | cluster | no |
| Open5GS | v2.7.7 (git tag) | docker/open5gs | **yes**: built from source and run with UERANSIM by open5gs/local_check.sh |
| UERANSIM | v3.3.0 (git tag) | docker/ueransim | **yes** (same check) |
| srsRAN Project (gNB) | release_25_10 (git tag) | docker/srsran-gnb | no -- check `cu_up.upf.bind_addr` and the ZMQ keys against the release's config reference |
| srsRAN 4G (srsUE) | release_23_11 (git tag) | docker/srsue | no |
| OpenAirInterface | OAI_TAG=2025.w10 (set to an existing tag before use) | oai/, k8s/ran-oai | no (optional profile) |
| MongoDB | mongo:6.0 | k8s/open5gs/mongodb.yaml | local check uses conda-forge mongodb |
| Multus CNI | v4.1.x thick plugin | host/02-cni.sh | no |
| OVS-CNI | v0.38.x | host/02-cni.sh | no |
| CNI reference plugins | v1.6.x (for `static` IPAM) | host/02-cni.sh | no |
| kube-prometheus-stack | chart 66.x | host/03-addons.sh | no |
| Chaos Mesh | chart 2.7.x | host/03-addons.sh | no |
| Mininet + Open vSwitch | Ubuntu packages | sdn/ | see sdn/README.md |
| Ryu | 4.34 with eventlet 0.30.2, Python 3.9 | sdn/ryu_apps | see sdn/README.md |
| ns-3 / 5G-LENA | ns-3.42 + nr-v3.3 (or newer pair listed as compatible by 5G-LENA) | ns3/ | no |
| Python (analysis) | 3.10-3.13, see pyproject.toml | testbed/ | yes (CI) |

"Validated" means exercised by an automated check in this repository. Everything else must be
validated on the real machine during Phase 1-2 and the result written in LOG.md.
