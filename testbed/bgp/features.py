"""Per-minute BGP features from MRT UPDATE messages, in the canonical long format.

Feature definitions follow the volume and AS-path feature families used in the BGP anomaly
literature (e.g. Al-Rousan and Trajkovic, "Machine learning models for classification of BGP
anomalies", HPSR 2012). Check each definition against the paper you compare with; the code
names every feature explicitly so a mismatch is easy to spot.

State is kept per (peer, prefix): the last announced AS path, or None after a withdrawal.
  announcements        announced prefixes (NLRI + MP_REACH)
  withdrawals          withdrawn prefixes (withdrawn routes + MP_UNREACH)
  updates              UPDATE messages
  unique_prefixes      distinct prefixes announced in the minute
  dup_announcements    announcement with the SAME path as the current one for (peer, prefix)
  implicit_withdrawals announcement with a DIFFERENT path while the prefix is still announced
  dup_withdrawals      withdrawal of a prefix that is not currently announced
  new_announcements    announcement of a prefix that is not currently announced
  avg/max_path_len     AS-path length (AS_SEQUENCE counted per AS, AS_SET counted as 1)
  avg_unique_path_len  path length after removing prepending (consecutive duplicate ASes)
  avg/max_edit_dist    edit distance between old and new path, on implicit withdrawals
  origin_igp/egp/incomplete   announcements by ORIGIN attribute
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURES = ["announcements", "withdrawals", "updates", "unique_prefixes", "dup_announcements",
            "implicit_withdrawals", "dup_withdrawals", "new_announcements", "avg_path_len",
            "max_path_len", "avg_unique_path_len", "avg_edit_dist", "max_edit_dist",
            "origin_igp", "origin_egp", "origin_incomplete"]


def _code(d) -> int:
    return int(next(iter(d))) if isinstance(d, dict) else int(d)


def edit_distance(a: tuple, b: tuple) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _path(attrs) -> tuple[tuple, int | None]:
    path: list = []
    origin = None
    for a in attrs:
        t = _code(a["type"])
        if t == 1:
            origin = _code(a["value"])
        elif t in (2, 17):          # AS_PATH, AS4_PATH (AS4_PATH overrides AS_TRANS segments)
            p = []
            for seg in a["value"]:
                st = _code(seg["type"])
                if st == 2:          # AS_SEQUENCE
                    p.extend(int(x) for x in seg["value"])
                elif st == 1:        # AS_SET counts as one hop
                    p.append(tuple(sorted(int(x) for x in seg["value"])))
            if t == 2 or p:
                path = p
    return tuple(path), origin


def _prefixes(lst) -> list[str]:
    return [f"{p['prefix']}/{p['length']}" for p in lst or []]


def _mp(attrs, kind: int) -> list[str]:
    out = []
    for a in attrs:
        if _code(a["type"]) == kind:
            v = a["value"]
            routes = v.get("nlri", []) if kind == 14 else v.get("withdrawn_routes", [])
            out += _prefixes(routes)
    return out


class Accumulator:
    def __init__(self):
        self.rib: dict[tuple[str, str], tuple | None] = {}
        self.bins: dict[int, dict] = defaultdict(lambda: defaultdict(list))

    def add(self, ts: int, peer: str, msg: dict) -> None:
        if _code(msg.get("type", {0: ""})) != 2:
            return
        b = self.bins[ts - ts % 60]
        b["updates"].append(1)
        attrs = msg.get("path_attributes", [])
        withdrawn = _prefixes(msg.get("withdrawn_routes")) + _mp(attrs, 15)
        announced = _prefixes(msg.get("nlri")) + _mp(attrs, 14)
        for pfx in withdrawn:
            key = (peer, pfx)
            b["withdrawals"].append(1)
            if self.rib.get(key) is None:
                b["dup_withdrawals"].append(1)
            self.rib[key] = None
        if not announced:
            return
        path, origin = _path(attrs)
        uniq = tuple(x for i, x in enumerate(path) if i == 0 or x != path[i - 1])
        for pfx in announced:
            key = (peer, pfx)
            old = self.rib.get(key)
            b["announcements"].append(1)
            b["prefix_set"].append(pfx)
            b["path_len"].append(len(path))
            b["unique_path_len"].append(len(uniq))
            if origin is not None:
                b[("origin_igp", "origin_egp", "origin_incomplete")[min(origin, 2)]].append(1)
            if old is None:
                b["new_announcements"].append(1)
            elif old == path:
                b["dup_announcements"].append(1)
            else:
                b["implicit_withdrawals"].append(1)
                b["edit_dist"].append(edit_distance(old, path))
            self.rib[key] = path

    def table(self) -> pd.DataFrame:
        rows = []
        for t, b in sorted(self.bins.items()):
            r = {"ts_s": t}
            for k in ("announcements", "withdrawals", "updates", "dup_announcements",
                      "implicit_withdrawals", "dup_withdrawals", "new_announcements",
                      "origin_igp", "origin_egp", "origin_incomplete"):
                r[k] = float(len(b.get(k, [])))
            r["unique_prefixes"] = float(len(set(b.get("prefix_set", []))))
            pl, upl, ed = b.get("path_len", []), b.get("unique_path_len", []), b.get("edit_dist", [])
            r["avg_path_len"] = float(np.mean(pl)) if pl else np.nan
            r["max_path_len"] = float(np.max(pl)) if pl else np.nan
            r["avg_unique_path_len"] = float(np.mean(upl)) if upl else np.nan
            r["avg_edit_dist"] = float(np.mean(ed)) if ed else 0.0
            r["max_edit_dist"] = float(np.max(ed)) if ed else 0.0
            rows.append(r)
        df = pd.DataFrame(rows, columns=["ts_s", *FEATURES])
        if len(df):   # minutes with no update at all are real zeros, not missing data
            full = pd.DataFrame({"ts_s": np.arange(df.ts_s.min(), df.ts_s.max() + 60, 60)})
            df = full.merge(df, on="ts_s", how="left")
            cnt = [c for c in FEATURES if not c.startswith(("avg_path", "max_path", "avg_unique"))]
            df[cnt] = df[cnt].fillna(0.0)
        return df


def features_from_files(files: list[Path], peer_filter: str | None = None) -> pd.DataFrame:
    import mrtparse
    acc = Accumulator()
    n_msg = 0
    for f in sorted(files):
        for e in mrtparse.Reader(str(f)):
            d = e.data
            if _code(d.get("type", {0: ""})) not in (16, 17) or "bgp_message" not in d:
                continue
            peer = str(d.get("peer_as", "")) + "@" + str(d.get("peer_ip", ""))
            if peer_filter and not peer.startswith(str(peer_filter) + "@"):
                continue
            ts = _code(d["timestamp"])
            acc.add(ts, peer, d["bgp_message"])
            n_msg += 1
    log.info("parsed %d BGP4MP messages from %d files", n_msg, len(files))
    return acc.table()


def to_canonical(df: pd.DataFrame, run_id: str, entity: str) -> pd.DataFrame:
    long = df.melt(id_vars="ts_s", var_name="metric", value_name="value")
    return long.assign(ts=(long["ts_s"] * 1000).astype("int64"), run_id=run_id, source="bgp",
                       entity=entity).drop(columns="ts_s")
