"""Download BGP UPDATE archives (MRT) from RIPE RIS and Route Views.

File names are taken from each month's directory index rather than generated, because the
update-file interval changed over the years (15 min in early RIS archives, 5 min later).
Every downloaded file is recorded with its size and SHA-256 in download_manifest.csv.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

RIS = "https://data.ris.ripe.net/{rrc}/{ym}/"
ROUTEVIEWS = "https://archive.routeviews.org/{coll}/bgpdata/{ym}/UPDATES/"
ROUTEVIEWS_RV1 = "https://archive.routeviews.org/bgpdata/{ym}/UPDATES/"   # the original route-views
_FILE_RE = re.compile(r'href="(updates\.(\d{8})\.(\d{4})\.(?:gz|bz2))"')


def list_month(base_url: str, session=None) -> list[tuple[str, dt.datetime]]:
    r = (session or requests).get(base_url, timeout=60)
    r.raise_for_status()
    out = []
    for name, ymd, hm in _FILE_RE.findall(r.text):
        t = dt.datetime.strptime(ymd + hm, "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
        out.append((name, t))
    return sorted(set(out), key=lambda x: x[1])


def month_urls(collector: str, kind: str, day: dt.date) -> str:
    ym = f"{day:%Y.%m}"
    if kind == "ripe_ris":
        return RIS.format(rrc=collector, ym=ym)
    if collector in ("route-views", "route-views1"):
        return ROUTEVIEWS_RV1.format(ym=ym)
    return ROUTEVIEWS.format(coll=collector, ym=ym)


def download_event(event: dict, collectors: dict, out_dir: Path, session=None) -> pd.DataFrame:
    sess = session or requests.Session()
    days = [dt.date.fromisoformat(d) for d in event["days"]]
    rows = []
    for kind, names in collectors.items():
        for coll in names or []:
            months = sorted({(d.year, d.month) for d in days})
            for (y, m) in months:
                base = month_urls(coll, kind, dt.date(y, m, 1))
                for name, t in list_month(base, sess):
                    if t.date() not in days:
                        continue
                    dst = out_dir / event["name"] / coll / name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        r = sess.get(base + name, timeout=300)
                        r.raise_for_status()
                        dst.write_bytes(r.content)
                    rows.append({"event": event["name"], "collector": coll, "file": str(dst),
                                 "file_time": t.isoformat(), "bytes": dst.stat().st_size,
                                 "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
            log.info("%s/%s: %d files", event["name"], coll, sum(r["collector"] == coll for r in rows))
    return pd.DataFrame(rows)
