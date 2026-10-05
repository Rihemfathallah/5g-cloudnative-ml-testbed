# SDN transport for N3

Four Open vSwitch switches in a ring, built by Mininet (`mininet_service.py`), controlled by Ryu
(`ryu_apps/ring_router.py`). The gNB's N3 interface attaches to **s1**, the UPF's to **s3** (OVS-CNI,
`infra/k8s/nad/n3.yaml`).

```
        s1  (gNB, 10.100.3.10)          forwarding tree (all links up):  s1-s2, s2-s3, s1-s4
       /  \                              N3 path:                         s1 -> s2 -> s3
     s4    s2                            s3-s4 is outside the tree (carries nothing)
       \  /
        s3  (UPF, 10.100.3.20)          s1-s2 or s2-s3 down -> tree s1-s4, s4-s3 (+ the other link)
```

## Why not spanning tree

Ryu's `stplib` was the first design. `check_ring.py` showed two problems: (1) after a failure on the
active path, the bridge behind the failure kept its stale root port and the blocked port never moved
to FORWARD — a permanent blackout instead of a reroute; (2) every new port (each pod restart) spent
2 × 15 s in LISTEN/LEARN. The controller now computes the tree itself and recomputes it on every
port-status message. Measured with the userspace datapath: reroute ~0.01 s, restore ~0.01 s.

## Fault semantics

* `delay d ms`, `loss p %` are applied with netem on BOTH ends of a link: RTT grows by 2d,
  round-trip loss is 1 − (1 − p)². `labels.csv` records the one-way value.
* `link_down` causes a short interruption (flow flush + re-learn), then traffic continues on the
  other side of the ring. The symptom is mostly in the SDN telemetry (link counters, tree state),
  hence the `no_link_state` ablation in `configs/detection.yaml`.
* Only links on the active path (s1-s2, s2-s3) are fault targets; a test enforces it.

## Install (host)

```bash
sudo apt-get install mininet openvswitch-switch linux-modules-extra-$(uname -r)   # netem lives in -extra
# Ryu needs Python 3.9 and an old setuptools:
python3.9 -m venv ~/ryu-env && . ~/ryu-env/bin/activate
pip install "setuptools==58.0.4" wheel
pip install --no-build-isolation "ryu==4.34" "eventlet==0.30.2" prometheus_client
```

## Check

```bash
sudo RYU_MANAGER=~/ryu-env/bin/ryu-manager DATAPATH=kernel python3 infra/sdn/check_ring.py
```

Last run in the build sandbox (userspace datapath, no netem): 7 passed, 0 failed, delay check
skipped — see `check_ring_result.json`. On the host the delay check must pass too.
