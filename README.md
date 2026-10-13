# Cloud-Native 5G Testbed for Handover Prediction and Anomaly Detection

Rihem Fathallah — personal research project (ML for 5G/6G networks). **Status: in progress.**
No result in this repository is final until it appears in `results/` with a manifest, produced on
the testbed described below.

An emulated 5G standalone network (Open5GS core, srsRAN gNB + srsUE over a ZeroMQ virtual radio)
runs in a single-node Kubernetes cluster; its N3 user-plane traffic crosses an SDN transport (a
Mininet Open vSwitch ring under a Ryu controller). Faults are injected into the transport (delay,
loss, link failure) and into the network functions (CPU and memory stress, pod failures) to build a
labelled dataset. Isolation Forest and an LSTM autoencoder detect them against a z-score baseline;
the same detectors are applied to public BGP data. A second track trains an LSTM to predict
handovers from signal traces and compares it with a per-speed-tuned 3GPP A3 rule.

```
 control plane      run orchestrator ── traffic generator ── fault injector ──► labels.csv
                                │                 │                │ Chaos Mesh (pods)
 same host ┌──────────────────────────────────────┼────────────────┼─────────────────────────┐
           │ k3s   srsUE pod ⇄ZMQ⇄ gNB pod ──N2──► Open5GS core (AMF SMF UPF NRF AUSF UDM ...) │
           │        + probe        │ N3 (Multus/OVS-CNI)        ▲ N3                          │
           │                       ▼                            │                              │
           │ SDN   s1 ── s2 ── s3 (UPF)      Ryu ring_router: controller-computed tree,       │
           │        └─ s4 ─┘                 reroute on link failure; Mininet links (netem)   │
           └─────────────────────────────────────────────────────────────────────────────────┘
 observability      probe · kubelet/cAdvisor · node-exporter · Open5GS /metrics · Ryu ──► Prometheus (5 s)
 data & ML          exporter ─► raw.parquet ─► windows ─► z-score · IForest · LSTM-AE · RF ref ─► results/*.csv
 offline tracks     BGP archives (RIPE RIS / Route Views)      handover traces (TR 38.901, Python | ns-3)
```

Architecture rationale, decisions (ADRs) and risks: see `docs/` (architecture notes; add your
exported design doc there). The decisions that changed during implementation are in **"What the checks found"** below.

---

## What is verified, and how

| Component | Verified here | Evidence |
|---|---|---|
| Analysis pipeline (schema, labels, windows, splits, detectors, evaluator) | yes | `pytest` (46 tests), `tb smoke` |
| Open5GS v2.7.7 configs (10 NFs, NRF registration, N4/PFCP SMF↔UPF) | yes, built from source and run | `infra/open5gs/local_check.py` → `local_check_result.json` |
| Open5GS metric names used by the metric map (SMF, UPF) | yes | `infra/open5gs/validated_metrics.txt` |
| AMF / NGAP / UE registration / user plane | **no** — the build sandbox kernel has no SCTP | run `local_check.py` on the testbed host |
| SDN ring: forwarding tree, path s1-s2-s3, reroute on link failure, restore | yes, Mininet + OVS + Ryu 4.34 | `infra/sdn/check_ring.py` → `check_ring_result.json` |
| Delay / loss injection (netem) | **no** — sandbox kernel has no `sch_netem` | run `check_ring.py` on the host |
| TR 38.901 UMa path loss (Python) | yes, identical to ns-3 (0.000 dB, 18 points) | `tests/crosscheck_ns3.py` → `crosscheck_ns3_result.csv` |
| srsRAN gNB / srsUE images and configs | **no** | Phase 1 on the host |
| k3s, Multus, OVS-CNI, Chaos Mesh, kube-prometheus-stack | **no** | Phase 1–2 on the host |
| ns-3 trace program (`ns3/nr-rsrp-traces.cc`) | **no** (not compiled) | build in an ns-3 tree |
| OpenAirInterface profile | **no** (optional, time-boxed) | `infra/oai/README.md` |

### What the checks found (and what changed because of them)

1. **Open5GS Docker build**: `-Dmetrics_impl=prometheus` does not exist in v2.7.7 and `libidn-dev`
   was missing — both would have broken the image. Fixed in `infra/docker/open5gs/Dockerfile`.
2. **Metric names**: `fivegs_smffunction_sm_sessionnbr` (in some docs) does not exist in v2.7.7; the
   SMF exposes `pfcp_sessions_active` and `ues_active`. Metric map fixed; SMF/UPF names now verified.
3. **Spanning tree does not work for this ring.** With Ryu's `stplib`, a failure on the active path
   never moved the blocked port to FORWARD: a link failure was a permanent blackout, and every new
   pod port waited 30 s in LISTEN/LEARN. Replaced by `ring_router.py`: the controller computes a
   deterministic BFS tree rooted at s1 and recomputes it on every port-status change. Measured
   reroute: ~10 ms (userspace datapath).
4. **Fault placement**: with the gNB on s1 and the UPF on s4 (first design), N3 would use the
   direct s4–s1 link and transport faults on s1-s2 / s2-s3 would hit idle links — labelled
   anomalies with no symptom. The UPF attaches to **s3**, so N3 follows s1-s2-s3, and transport
   faults target only those two links (enforced by a test).
5. **Ryu packaging**: Ryu 4.34 needs `setuptools==58` and `--no-build-isolation`, and
   `prometheus_client.start_http_server` fails under eventlet — metrics are served through Ryu's
   own WSGI server.
6. **Handover channel**: hard LOS/NLOS switching made RSRP jump 35–40 dB between two samples 3 cm
   apart, causing failures no policy could avoid. Now: a smooth (Gaussian-correlated) LOS field and a
   soft LOS transition of a few metres. 99th-percentile RSRP step at 120 km/h: 3.8 dB.
7. **A3 tuning grid**: the 120 km/h optimum sat on the grid edge (6 dB). Grid extended to 10 dB and the
   pipeline now warns whenever an optimum sits on an edge (an untuned baseline biases the comparison).

---

## Quick start (analysis only, any machine)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,bgp]"
pytest                       # 46 tests
tb smoke --fast              # synthetic end-to-end check -> results/smoke (NEVER a result)
tb handover all --fast -o results/smoke_ho
```

`tb smoke` replaces the 5G network with `testbed/sim/synthetic_testbed.py`, a statistical stand-in,
to exercise every component. Its numbers say nothing about 5G; figure scripts refuse to plot them
without `--allow-synthetic` and stamp them "SYNTHETIC — NOT A RESULT".

## Full testbed (Ubuntu 22.04/24.04, 16 GB RAM recommended)

```bash
infra/host/00-prereqs.sh            # mininet, OVS, iperf3, sctp + netem modules, docker, helm
infra/host/01-k3s.sh                # k3s with the static CPU manager (2 CPUs reserved)
infra/host/02-cni.sh                # Multus + OVS-CNI + static IPAM
infra/host/03-addons.sh             # kube-prometheus-stack (5 s scrape) + Chaos Mesh
infra/host/04-images.sh             # builds and imports all images (srsRAN takes a while)

# validate before Kubernetes (both must pass on the host):
sudo OPEN5GS_BIN=... UERANSIM_BIN=... python3 infra/open5gs/local_check.py
sudo RYU_MANAGER=... python3 infra/sdn/check_ring.py

# bring up, in this order:
ryu-manager infra/sdn/ryu_apps/ring_router.py &          # Python 3.9 env, see infra/sdn/README.md
sudo python3 infra/sdn/mininet_service.py &
infra/host/up.sh k8s-srsran                               # or k8s-ueransim on 8 GB machines

tb plan configs/runs/k8s_srsran_mixed_a.yaml              # inspect the fault plan
tb run  configs/runs/k8s_srsran_mixed_a.yaml              # preflight -> 10 min warm-up -> 5 h -> export
```

A 5 h run holds ~11 faults; with 7 fault types × 2 severities, collect **at least 4 runs**
(change `seed` and `name`) before per-type results mean anything.

## Experiments → commands

| Experiment | Command |
|---|---|
| Windows from runs | `tb windows runs/<id1> runs/<id2> ... -o data/main` |
| RQ1 detection (all models, 5 seeds) | `tb detect data/main -o results/detection_main` |
| Ablation without SDN link-state columns | `tb windows ... -o data/no_link_state --exclude 'link_forwarding$' 'link_up$' 'tree_change_rate$'` |
| RAN transfer (train srsRAN, test OAI/UERANSIM run) | `tb windows runs/<srsran...> --test-only runs/<oai> -o data/transfer` |
| RQ1b BGP | set `verified: true` in `configs/bgp.yaml` after checking intervals, then `tb bgp all` |
| RQ2 handover (Python traces) | `tb handover all -o results/handover` |
| RQ2 with ns-3 traces | `NS3_DIR=~/ns-3-dev ns3/run_all.sh`, set `trace_source: ns3`, `tb handover all` |
| Figures | `figures/make_all.sh results/detection_main results/handover` |

## Repository

```
configs/        run manifests, metric map, detection / handover / bgp configs
testbed/        control (plan, inject, run), data (export, windows), anomaly, handover, bgp, sim (synthetic)
infra/          host scripts, Dockerfiles, k8s manifests (Open5GS generated by infra/open5gs/render.py),
                sdn (Mininet service, Ryu app, check), open5gs (render, provision, local check)
ns3/            ns-3 TR 38.901 trace program + driver
figures/        one script per figure, CSV in -> PDF/PNG out
tests/          pytest suite + ns-3 cross-check
report/         IEEE-format report skeleton
runs/ results/  data and results (git-ignored; manifests are kept)
```

## Ground rules

Only real results; Git from day one; YAML configs, fixed seeds, CSV results, one script per figure;
always compare against tuned baselines; a `LOG.md` entry per run. A report number must trace back
to a `results/*.csv` and a run manifest with `git.dirty: false`.

## CV wording

Keep "In progress" until figures exist. Tools line: list ns-3 only if `ns3/run_all.sh` produced the
traces you report — the program uses ns-3's TR 38.901 channel models (the ones 5G-LENA builds on),
not the 5G-LENA NR stack, so "ns-3 (3GPP TR 38.901 channel models)" is the accurate wording.
Remove OpenAirInterface if its profile never registers a UE.
