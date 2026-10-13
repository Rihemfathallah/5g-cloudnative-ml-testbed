# Lab log

One entry per working session and per run. Newest first. Record what changed, what broke, the
run id, and anything that makes a run unusable (it is then excluded from the report, not deleted).


## Template

```
### YYYY-MM-DD
- run: <run_id>  (config: configs/runs/<name>.yaml, git <short hash>, dirty: no)
- state: preflight PASS | FAIL (<which check>)
- faults: <n ok> / <n failed>; notable: ...
- problems / decisions: ...
- usable for report: yes | no (<why>)
```

## 2026-11-07 — implementation and checks in the build sandbox

- Repository implemented end to end; 46 tests pass; synthetic smoke pipeline runs.
- Open5GS v2.7.7 built from source; `local_check.py`: 18 passed, 0 failed, 8 skipped (no SCTP in
  the sandbox kernel: AMF / NGAP / UE / user plane must be checked on the host).
- Docker build fixes: removed non-existent meson option `metrics_impl`, added `libidn-dev`.
- Metric map: `fivegs_smffunction_sm_sessionnbr` does not exist in v2.7.7 -> `pfcp_sessions_active`, `ues_active`.
- SDN: Ryu stplib never reconverged after a failure on the active path -> replaced by a controller-
  computed BFS tree (`ring_router.py`). `check_ring.py`: 7 passed, reroute ~0.01 s. UPF moved to s3.
- Handover: hard LOS switching gave 35-40 dB jumps per 40 ms sample -> smooth LOS field + soft LOS.
  Python TR 38.901 UMa path loss identical to ns-3 (tests/crosscheck_ns3.py).
- A3 grid extended to 10 dB hysteresis (optimum was on the 6 dB edge at 120 km/h).
