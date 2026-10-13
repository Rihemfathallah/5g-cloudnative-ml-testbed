"""Values that must agree across files. Each of these, if wrong, gives a testbed that starts but
does not work: the UE never registers, N3 bypasses the SDN ring, or a fault hits an idle link."""
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra/open5gs"))
import render  # noqa: E402


def _read(p):
    return (ROOT / p).read_text()


def test_generated_manifests_are_up_to_date(tmp_path):
    current = _read("infra/k8s/open5gs/generated.yaml")
    subprocess.run([sys.executable, str(ROOT / "infra/open5gs/render.py"), "k8s"], check=True, capture_output=True)
    assert _read("infra/k8s/open5gs/generated.yaml") == current, "run infra/open5gs/render.py k8s and commit"


def test_n3_addresses_agree():
    nad = list(yaml.safe_load_all(_read("infra/k8s/nad/n3.yaml")))
    cfg = {d["metadata"]["name"]: json.loads(d["spec"]["config"]) for d in nad}
    assert cfg["n3-core"]["ipam"]["addresses"][0]["address"] == f"{render.N3_UPF}/24"
    assert cfg["n3-core"]["bridge"] == "s3" and cfg["n3-ran"]["bridge"] == "s1"
    gnb_ip = cfg["n3-ran"]["ipam"]["addresses"][0]["address"].split("/")[0]
    assert gnb_ip == "10.100.3.10"
    assert f'value: "{gnb_ip}"' in _read("infra/k8s/ran-srsran/ran.yaml")
    assert gnb_ip in _read("infra/k8s/ran-ueransim/ran.yaml")
    upf = render.nf_config("upf", "k8s", "/x")["upf"]
    assert upf["gtpu"]["server"][0]["address"] == render.N3_UPF


def test_credentials_and_plmn_agree():
    sub = json.loads(_read("infra/open5gs/subscriber.json"))
    ue_conf = _read("infra/srsran/ue.conf.tmpl")
    ue_yaml = yaml.safe_load(_read("infra/ueransim/ue.yaml.tmpl"))
    assert f"imsi = {sub['imsi']}" in ue_conf and ue_yaml["supi"] == f"imsi-{sub['imsi']}"
    assert sub["security"]["k"].lower() in ue_conf.lower() and ue_yaml["key"].lower() == sub["security"]["k"].lower()
    assert sub["security"]["opc"].lower() in ue_conf.lower() and ue_yaml["op"].lower() == sub["security"]["opc"].lower()
    plmn = render.PLMN["mcc"] + render.PLMN["mnc"]
    gnb = _read("infra/srsran/gnb.yaml.tmpl")
    assert f'plmn: "{plmn}"' in gnb and f"tac: {render.TAC}" in gnb
    gyaml = yaml.safe_load(_read("infra/ueransim/gnb.yaml.tmpl").replace("__", "x"))
    assert gyaml["mcc"] == render.PLMN["mcc"] and gyaml["tac"] == render.TAC
    dnn = sub["slice"][0]["session"][0]["name"]
    assert f"apn = {dnn}" in ue_conf and ue_yaml["sessions"][0]["apn"] == dnn


def test_sdn_port_maps_agree():
    mn = _read("infra/sdn/mininet_service.py")
    ryu = _read("infra/sdn/ryu_apps/ring_router.py")
    ring = re.search(r"RING = (\[.*?\])", mn).group(1)
    assert ring == '[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s1")]'
    assert "port1=1, port2=2" in mn                    # clockwise end = port 1
    links = re.search(r"LINKS = (\{.*?\})\n", ryu, re.S).group(1)
    for name, (a, b) in zip(["s1-s2", "s2-s3", "s3-s4", "s4-s1"], [(1, 2), (2, 3), (3, 4), (4, 1)]):
        assert f'"{name}": (({a}, 1), ({b}, 2))' in links
    mm = yaml.safe_load(_read("configs/metric_map.yaml"))
    sdn = [g for g in mm["groups"] if g["metric"] == "link_tx_bytes_rate"][0]
    assert sdn["entities"] == ["s1-s2", "s2-s3", "s3-s4", "s4-s1"]


def test_bfs_tree_is_the_documented_path():
    sys.path.insert(0, str(ROOT / "infra/sdn/ryu_apps"))
    src = _read("infra/sdn/ryu_apps/ring_router.py")
    ns = {"deque": __import__("collections").deque}
    exec(src[src.index("ROOT = 1"):src.index("class _Collector")], ns)
    tree = ns["bfs_tree"]
    assert tree({"s1-s2", "s2-s3", "s3-s4", "s4-s1"}) == {"s1-s2", "s2-s3", "s4-s1"}
    assert tree({"s2-s3", "s3-s4", "s4-s1"}) == {"s4-s1", "s3-s4", "s2-s3"}     # s1-s2 down
    assert tree({"s1-s2", "s3-s4", "s4-s1"}) == {"s1-s2", "s4-s1", "s3-s4"}     # s2-s3 down


def test_transport_faults_target_the_active_path():
    """Faults on links outside the forwarding tree would be labelled anomalies with no symptom."""
    for p in (ROOT / "configs/runs").glob("*.yaml"):
        cfg = yaml.safe_load(p.read_text())
        for c in cfg["faults"]["catalog"]:
            if c["injector"] == "mininet":
                assert set(c["targets"]) <= {"link:s1-s2", "link:s2-s3"}, (p.name, c["type"])


def test_metric_map_expressions_aggregate():
    mm = yaml.safe_load(_read("configs/metric_map.yaml"))
    for g in mm["groups"]:
        q = g["promql"].strip()
        assert re.match(r"^(1 - )?(sum|max|min|avg)\(", q), f"{g['metric']} must aggregate to one series"


def test_pods_that_need_isolation_are_guaranteed():
    upf = render.deployment("upf")["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert upf["requests"] == upf["limits"] and upf["requests"]["cpu"].isdigit()
    ran = list(yaml.safe_load_all(_read("infra/k8s/ran-srsran/ran.yaml")))
    for d in (x for x in ran if x["kind"] == "Deployment"):
        for c in d["spec"]["template"]["spec"]["containers"]:
            assert c["resources"]["requests"] == c["resources"]["limits"], (d["metadata"]["name"], c["name"])


def test_k8s_manifests_parse_and_selectors_match():
    n = 0
    for p in (ROOT / "infra/k8s").rglob("*.yaml"):
        if p.name.startswith("values-") or p.name == "kustomization.yaml":
            continue
        text = p.read_text().replace("${OAI_TAG}", "tag")
        for d in yaml.safe_load_all(text):
            if not d:
                continue
            n += 1
            assert "apiVersion" in d and "kind" in d, p
            if d["kind"] == "Deployment":
                sel = d["spec"]["selector"]["matchLabels"]
                labels = d["spec"]["template"]["metadata"]["labels"]
                assert sel.items() <= labels.items(), (p.name, d["metadata"]["name"])
                assert d["metadata"]["namespace"] == "5g"
    assert n >= 30


def test_kustomization_files_exist():
    for k in (ROOT / "infra/k8s").rglob("kustomization.yaml"):
        doc = yaml.safe_load(k.read_text())
        for r in doc.get("resources", []):
            assert (k.parent / r).exists(), (k, r)
        for g in doc.get("configMapGenerator", []):
            for f in g.get("files", []):
                if "generated/" in f:          # created by infra/oai/fetch-configs.sh
                    continue
                assert (k.parent / f).exists(), (k, f)
