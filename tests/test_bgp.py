"""BGP features on a hand-built MRT file (BGP4MP_MESSAGE_AS4)."""
import ipaddress
import struct

import pytest

mrtparse = pytest.importorskip("mrtparse")

from testbed.bgp.features import edit_distance, features_from_files, to_canonical  # noqa: E402
from testbed.bgp.pipeline import event_labels  # noqa: E402


def _attr(t, val):
    return struct.pack("!BBB", 0x40, t, len(val)) + val


def _pfx(p, l):
    return bytes([l]) + ipaddress.ip_network(f"{p}/{l}").network_address.packed[:(l + 7) // 8]


def _update(withdrawn=(), path=(), nlri=(), origin=0):
    w = b"".join(_pfx(p, l) for p, l in withdrawn)
    a = b""
    if nlri:
        a = (_attr(1, bytes([origin])) +
             _attr(2, struct.pack("!BB", 2, len(path)) + b"".join(struct.pack("!I", x) for x in path)) +
             _attr(3, bytes([10, 0, 0, 1])))
    n = b"".join(_pfx(p, l) for p, l in nlri)
    body = struct.pack("!H", len(w)) + w + struct.pack("!H", len(a)) + a + n
    return b"\xff" * 16 + struct.pack("!HB", 19 + len(body), 2) + body


def _mrt(ts, msg, peer_as=65001):
    b = struct.pack("!IIHH", peer_as, 65000, 0, 1) + bytes([192, 0, 2, 1, 192, 0, 2, 2]) + msg
    return struct.pack("!IHHI", ts, 16, 4, len(b)) + b


def test_edit_distance():
    assert edit_distance((1, 2, 3), (1, 2, 3)) == 0
    assert edit_distance((1, 2, 3), (1, 4, 3)) == 1
    assert edit_distance((1, 2), (1, 2, 3, 4)) == 2


def test_features(tmp_path):
    t = 1043452800                         # 2003-01-25 00:00 UTC, minute aligned
    p = ("8.8.8.0", 24)
    data = (_mrt(t + 1, _update(path=(65001, 3356, 15169), nlri=[p])) +          # new
            _mrt(t + 2, _update(path=(65001, 3356, 15169), nlri=[p])) +          # duplicate
            _mrt(t + 3, _update(path=(65001, 174, 174, 15169), nlri=[p])) +      # implicit withdrawal, prepending
            _mrt(t + 61, _update(withdrawn=[p])) +                              # withdrawal (next minute)
            _mrt(t + 62, _update(withdrawn=[p])) +                              # duplicate withdrawal
            _mrt(t + 185, _update(path=(65001, 15169), nlri=[p], origin=2)))     # minute 3, incomplete
    f = tmp_path / "updates.20030125.0000"
    f.write_bytes(data)
    df = features_from_files([f]).set_index("ts_s")
    m0, m1, m2, m3 = (df.loc[t + 60 * k] for k in range(4))
    assert (m0.announcements, m0.new_announcements, m0.dup_announcements, m0.implicit_withdrawals) == (3, 1, 1, 1)
    assert m0.max_path_len == 4 and m0.avg_unique_path_len == pytest.approx((3 + 3 + 3) / 3)
    assert m0.avg_edit_dist == 2.0 and m0.origin_igp == 3          # 1 substitution + 1 insertion
    assert (m1.withdrawals, m1.dup_withdrawals, m1.announcements) == (2, 1, 0)
    assert m2.updates == 0 and m2.announcements == 0          # empty minute = zeros, not missing
    assert m3.origin_incomplete == 1 and m3.new_announcements == 1
    long = to_canonical(df.reset_index(), "bgp_test", "bgp")
    assert set(long.columns) >= {"ts", "metric", "value", "entity", "run_id", "source"}


def test_event_labels():
    ev = {"name": "x", "anomalous": [["2003-01-25T05:30:00Z", "2003-01-25T06:00:00Z"]]}
    lab = event_labels(ev)
    assert lab.iloc[0]["t_removed"] - lab.iloc[0]["t_applied"] == 30 * 60 * 1000
