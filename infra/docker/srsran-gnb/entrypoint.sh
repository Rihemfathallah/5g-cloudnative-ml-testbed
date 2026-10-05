#!/bin/sh
# Resolve peers at start (Service DNS -> ClusterIP), render the config, start the gNB.
set -eu
resolve() { getent hosts "$1" | awk '{print $1; exit}'; }
AMF_IP=$(resolve "${AMF_HOST:-amf.5g.svc.cluster.local}")
UE_IP=$(resolve "${UE_HOST:-ue-zmq.5g.svc.cluster.local}")
N3_IP=${N3_IP:-10.100.3.10}
# The N3 interface must exist and carry the expected address, otherwise GTP-U would leave
# through eth0 and bypass the SDN ring. Fail loudly instead.
ip -4 addr show dev n3 | grep -q "inet ${N3_IP}/" || { echo "n3 interface missing or wrong address"; exit 1; }
sed -e "s/__AMF_IP__/${AMF_IP}/" -e "s/__POD_IP__/${POD_IP}/" -e "s/__N3_IP__/${N3_IP}/" \
    -e "s/__UE_IP__/${UE_IP}/" /etc/srsran/gnb.yaml.tmpl > /tmp/gnb.yaml
echo "AMF=${AMF_IP} UE=${UE_IP} N2=${POD_IP} N3=${N3_IP}"
exec gnb -c /tmp/gnb.yaml
