"""Expand configs/metric_map.yaml into concrete (entity, metric, promql) queries."""
from __future__ import annotations

import dataclasses as dc

from testbed.common.config import load_yaml


@dc.dataclass(frozen=True)
class Query:
    entity: str
    metric: str
    promql: str
    unit: str = ""
    verify: bool = False


def load_metric_map(path) -> tuple[list[Query], dict]:
    doc = load_yaml(path)
    out: list[Query] = []
    seen: set[tuple[str, str]] = set()
    for g in doc["groups"]:
        for ent in g["entities"]:
            key = (ent, g["metric"])
            if key in seen:
                raise ValueError(f"duplicate canonical pair {key} in metric map")
            seen.add(key)
            out.append(Query(ent, g["metric"], g["promql"].replace("{entity}", ent),
                             g.get("unit", ""), bool(g.get("verify", False))))
    return out, {k: v for k, v in doc.items() if k != "groups"}
