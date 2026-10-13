#!/usr/bin/env python3
"""Cross-check testbed/handover/channel.py against ns-3's own TR 38.901 UMa model.

Uses the ns-3 Python bindings (`pip install ns3`). Shadowing is disabled and the channel
condition is forced (always-LOS / never-LOS), so both sides must agree on deterministic path
loss to within numerical precision. Writes tests/crosscheck_ns3_result.csv.

    python3 tests/crosscheck_ns3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testbed.handover.channel import pathloss_uma_los, pathloss_uma_nlos  # noqa: E402

FC, HBS, HUT = 3.5e9, 25.0, 1.5


def ns3_pathloss(d2d: float, los: bool) -> float:
    from ns import ns
    cond = (ns.CreateObject[ns.AlwaysLosChannelConditionModel]() if los
            else ns.CreateObject[ns.NeverLosChannelConditionModel]())
    m = ns.CreateObject[ns.ThreeGppUmaPropagationLossModel]()
    m.SetAttribute("Frequency", ns.DoubleValue(FC))
    m.SetAttribute("ShadowingEnabled", ns.BooleanValue(False))
    m.SetChannelConditionModel(cond)
    a = ns.CreateObject[ns.ConstantPositionMobilityModel]()
    b = ns.CreateObject[ns.ConstantPositionMobilityModel]()
    a.SetPosition(ns.Vector(0, 0, HBS))
    b.SetPosition(ns.Vector(d2d, 0, HUT))
    return -m.CalcRxPower(0.0, b, a)


def main() -> int:
    rows = []
    for d in [20, 50, 100, 200, 300, 500, 800, 1200, 2000]:
        for los in (True, False):
            ours = float((pathloss_uma_los if los else pathloss_uma_nlos)(np.array(d), HBS, HUT, FC / 1e9))
            ref = ns3_pathloss(d, los)
            rows.append({"d2d_m": d, "los": los, "ours_db": round(ours, 3), "ns3_db": round(ref, 3),
                         "diff_db": round(ours - ref, 4)})
    df = pd.DataFrame(rows)
    out = Path(__file__).with_name("crosscheck_ns3_result.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    worst = df["diff_db"].abs().max()
    print(f"\nmax |difference| = {worst:.4f} dB -> {'PASS' if worst < 0.05 else 'FAIL'}")
    return 0 if worst < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
