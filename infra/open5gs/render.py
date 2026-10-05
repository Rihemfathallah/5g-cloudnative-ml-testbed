#!/usr/bin/env python3
"""Single source of truth for the Open5GS core.

  python3 infra/open5gs/render.py k8s     -> infra/k8s/open5gs/*.yaml   (ConfigMap + Deployments + Services)
  python3 infra/open5gs/render.py local   -> build/open5gs-local/*.yaml (127.0.0.x, used by
                                              infra/open5gs/local_check.sh to validate the configs
                                              by actually running Open5GS + UERANSIM on one host)

The NF configurations are built as Python dicts below and dumped as YAML, so the local check and
the cluster use the same structure; only addresses differ. In Kubernetes, `__POD_IP__` is replaced
at container start from the downward API (status.podIP), and every SBI server advertises its
Service name so other NFs reach it through cluster DNS.

Open5GS version: see infra/VERSIONS.md (configuration schema of v2.7.x).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


class _NoAlias(yaml.SafeDumper):
    """Write repeated dicts in full: no &id001 anchors in files that humans compare to docs."""
    def ignore_aliases(self, data):
        return True


def dump(obj, stream=None, all_docs=False):
    if all_docs:
        return yaml.dump_all(obj, stream, Dumper=_NoAlias, sort_keys=False)
    return yaml.dump(obj, stream, Dumper=_NoAlias, sort_keys=False)


ROOT = Path(__file__).resolve().parents[2]
NS = "5g"
PLMN = {"mcc": "001", "mnc": "01"}
TAC = 7
SST = 1
UE_SUBNET, UE_GW = "10.45.0.1/16", "10.45.0.1"
N3_UPF = "10.100.3.20"          # UPF address on the SDN ring (Multus n3-core, bridge s3)
IMAGE = "testbed/open5gs:2.7.7"
SBI_PORT, METRICS_PORT = 7777, 9090

# local profile: the addresses used by Open5GS's own sample configs
LOCAL_IP = {"nrf": "127.0.0.10", "ausf": "127.0.0.11", "udm": "127.0.0.12", "pcf": "127.0.0.13",
            "nssf": "127.0.0.14", "bsf": "127.0.0.15", "udr": "127.0.0.20", "amf": "127.0.0.5",
            "smf": "127.0.0.4", "upf": "127.0.0.7"}
NFS = ["nrf", "ausf", "udm", "udr", "pcf", "nssf", "bsf", "amf", "smf", "upf"]
WITH_METRICS = {"amf", "smf", "upf", "pcf"}


def addr(nf: str, profile: str) -> str:
    return LOCAL_IP[nf] if profile == "local" else "__POD_IP__"


def host(nf: str, profile: str) -> str:
    return LOCAL_IP[nf] if profile == "local" else f"{nf}.{NS}.svc.cluster.local"


def sbi(nf: str, profile: str) -> dict:
    srv = {"address": addr(nf, profile), "port": SBI_PORT}
    if profile != "local":
        srv["advertise"] = host(nf, profile)
    d = {"server": [srv]}
    if nf != "nrf":
        d["client"] = {"nrf": [{"uri": f"http://{host('nrf', profile)}:{SBI_PORT}"}]}
    return d


def metrics(nf: str, profile: str) -> dict:
    return {"server": [{"address": addr(nf, profile), "port": METRICS_PORT}]}


def nf_config(nf: str, profile: str, log_dir: str) -> dict:
    db = ("mongodb://127.0.0.1/open5gs" if profile == "local"
          else f"mongodb://mongodb.{NS}.svc.cluster.local/open5gs")
    base = {"logger": {"file": {"path": f"{log_dir}/{nf}.log"}, "level": "info"},
            "global": {"max": {"ue": 64}}}
    body: dict = {"sbi": sbi(nf, profile)}
    if nf in WITH_METRICS:
        body["metrics"] = metrics(nf, profile)
    if nf == "nrf":
        body["serving"] = [{"plmn_id": PLMN}]
    elif nf in ("udr", "pcf"):
        base["db_uri"] = db
    elif nf == "nssf":
        body["sbi"]["client"]["nsi"] = [{"uri": f"http://{host('nrf', profile)}:{SBI_PORT}",
                                         "s_nssai": {"sst": SST}}]
    elif nf == "amf":
        body.update({
            "ngap": {"server": [{"address": addr(nf, profile)}]},
            "guami": [{"plmn_id": PLMN, "amf_id": {"region": 2, "set": 1}}],
            "tai": [{"plmn_id": PLMN, "tac": TAC}],
            "plmn_support": [{"plmn_id": PLMN, "s_nssai": [{"sst": SST}]}],
            "security": {"integrity_order": ["NIA2", "NIA1", "NIA0"],
                         "ciphering_order": ["NEA0", "NEA1", "NEA2"]},
            "network_name": {"full": "Testbed", "short": "TB"},
            "amf_name": "testbed-amf0",
            "time": {"t3512": {"value": 540}},
        })
    elif nf == "smf":
        upf_pfcp = LOCAL_IP["upf"] if profile == "local" else host("upf", profile)
        body.update({
            "pfcp": {"server": [{"address": addr(nf, profile)}],
                     "client": {"upf": [{"address": upf_pfcp}]}},
            "gtpc": {"server": [{"address": addr(nf, profile)}]},
            "gtpu": {"server": [{"address": addr(nf, profile)}]},
            "session": [{"subnet": UE_SUBNET, "gateway": UE_GW}],
            "dns": ["8.8.8.8", "8.8.4.4"],
            "mtu": 1400,
        })
    elif nf == "upf":
        smf_pfcp = LOCAL_IP["smf"] if profile == "local" else host("smf", profile)
        body.update({
            "pfcp": {"server": [{"address": addr(nf, profile)}],
                     "client": {"smf": [{"address": smf_pfcp}]}},
            # GTP-U is bound to the SDN-ring address in the cluster (ADR-2): if it were bound to
            # the pod IP, N3 would silently bypass the Mininet switches.
            "gtpu": {"server": [{"address": LOCAL_IP["upf"] if profile == "local" else N3_UPF}]},
            "session": [{"subnet": UE_SUBNET, "gateway": UE_GW, "dev": "ogstun"}],
        })
    base[nf] = body
    return base


# ------------------------------------------------------------------------------- Kubernetes
def k8s_objects() -> list[dict]:
    cm = {"apiVersion": "v1", "kind": "ConfigMap",
          "metadata": {"name": "open5gs-config", "namespace": NS},
          "data": {f"{nf}.yaml": dump(nf_config(nf, "k8s", "/var/log/open5gs"))
                   for nf in NFS}}
    objs = [cm]
    for nf in NFS:
        objs += [deployment(nf), service(nf)]
    return objs


def ports(nf: str) -> list[dict]:
    p = [{"name": "sbi", "containerPort": SBI_PORT, "protocol": "TCP"}]
    if nf in WITH_METRICS:
        p.append({"name": "metrics", "containerPort": METRICS_PORT, "protocol": "TCP"})
    if nf == "amf":
        p.append({"name": "ngap", "containerPort": 38412, "protocol": "SCTP"})
    if nf in ("smf", "upf"):
        p.append({"name": "pfcp", "containerPort": 8805, "protocol": "UDP"})
    if nf == "upf":
        p.append({"name": "gtpu", "containerPort": 2152, "protocol": "UDP"})
    return p


UPF_INIT = """set -e
ip tuntap add name ogstun mode tun 2>/dev/null || true
ip addr replace 10.45.0.1/16 dev ogstun
ip link set ogstun up
sysctl -w net.ipv4.ip_forward=1
iptables -t nat -C POSTROUTING -s 10.45.0.0/16 ! -o ogstun -j MASQUERADE 2>/dev/null || \\
  iptables -t nat -A POSTROUTING -s 10.45.0.0/16 ! -o ogstun -j MASQUERADE
"""


def deployment(nf: str) -> dict:
    start = (f'sed "s/__POD_IP__/$POD_IP/g" /etc/open5gs-tmpl/{nf}.yaml > /tmp/{nf}.yaml && '
             f'mkdir -p /var/log/open5gs && exec open5gs-{nf}d -c /tmp/{nf}.yaml')
    if nf == "upf":
        start = UPF_INIT + start
    c = {"name": nf, "image": IMAGE, "imagePullPolicy": "Never",
         "command": ["/bin/sh", "-c", start],
         "env": [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}],
         "ports": ports(nf),
         "volumeMounts": [{"name": "config", "mountPath": "/etc/open5gs-tmpl"}],
         "readinessProbe": {"tcpSocket": {"port": SBI_PORT}, "periodSeconds": 2, "failureThreshold": 2},
         "resources": {"requests": {"cpu": "100m", "memory": "64Mi"}, "limits": {"memory": "512Mi"}}}
    pod = {"containers": [c], "volumes": [{"name": "config", "configMap": {"name": "open5gs-config"}}],
           "terminationGracePeriodSeconds": 5}
    meta = {"labels": {"app": nf, "part-of": "open5gs"}}
    if nf == "upf":
        # Guaranteed QoS with an integer CPU -> exclusive core under the static CPU manager, so a
        # StressChaos on the UPF cannot spill onto the radio emulation (ADR-4 / core isolation).
        c["resources"] = {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}}
        c["securityContext"] = {"privileged": True}
        c["volumeMounts"].append({"name": "tun", "mountPath": "/dev/net/tun"})
        pod["volumes"].append({"name": "tun", "hostPath": {"path": "/dev/net/tun", "type": "CharDevice"}})
        meta["annotations"] = {"k8s.v1.cni.cncf.io/networks": '[{"name": "n3-core", "interface": "n3"}]'}
        # data-network sidecar: iperf3 servers at the UE gateway, reached through the 5G tunnel
        pod["containers"].append({
            "name": "dn", "image": "testbed/probe:1.0", "imagePullPolicy": "Never",
            "command": ["/bin/sh", "-c", "iperf3 -s -p 5201 -D && exec iperf3 -s -p 5202"],
            # requests == limits keeps the POD Guaranteed; a non-integer CPU keeps this sidecar in
            # the shared pool, so only the UPF container gets the exclusive core
            "resources": {"requests": {"cpu": "500m", "memory": "128Mi"},
                          "limits": {"cpu": "500m", "memory": "128Mi"}}})
    return {"apiVersion": "apps/v1", "kind": "Deployment",
            "metadata": {"name": nf, "namespace": NS, "labels": {"app": nf}},
            "spec": {"replicas": 1, "selector": {"matchLabels": {"app": nf}},
                     "strategy": {"type": "Recreate"},
                     "template": {"metadata": meta, "spec": pod}}}


def service(nf: str) -> dict:
    sp = [{"name": p["name"], "port": p["containerPort"], "targetPort": p["containerPort"],
           "protocol": p["protocol"]} for p in ports(nf)]
    return {"apiVersion": "v1", "kind": "Service",
            "metadata": {"name": nf, "namespace": NS, "labels": {"app": nf}},
            "spec": {"selector": {"app": nf}, "ports": sp}}


def main(profile: str) -> None:
    if profile == "k8s":
        out = ROOT / "infra/k8s/open5gs/generated.yaml"
        with open(out, "w") as f:
            f.write("# GENERATED by infra/open5gs/render.py -- edit that file, not this one.\n")
            dump(k8s_objects(), f, all_docs=True)
        print(out)
    elif profile == "local":
        out = ROOT / "build/open5gs-local"
        out.mkdir(parents=True, exist_ok=True)
        for nf in NFS:
            (out / f"{nf}.yaml").write_text(dump(nf_config(nf, "local", str(out / "log"))))
        (out / "log").mkdir(exist_ok=True)
        print(out)
    else:
        raise SystemExit("usage: render.py k8s|local")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "k8s")
