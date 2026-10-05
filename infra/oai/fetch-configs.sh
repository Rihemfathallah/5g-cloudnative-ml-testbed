#!/usr/bin/env bash
# Download and patch the OAI RF-simulator gNB config for the pinned tag.
set -euo pipefail
TAG=${OAI_TAG:-$(grep -oP 'OAI_TAG=\K\S+' "$(dirname "$0")/../VERSIONS.md")}
BASE="https://gitlab.eurecom.fr/oai/openairinterface5g/-/raw/${TAG}/ci-scripts/conf_files"
OUT="$(dirname "$0")/generated"; mkdir -p "$OUT"
curl -fsSL "$BASE/gnb.sa.band78.106prb.rfsim.conf" -o "$OUT/gnb.conf.orig"
# Patch: PLMN, TAC, slice, AMF and interface addresses. Each sed prints a warning if it changed
# nothing, so an upstream format change is noticed instead of silently ignored.
patch() { local before; before=$(md5sum "$OUT/gnb.conf"); sed -i "$1" "$OUT/gnb.conf"; \
          [ "$before" != "$(md5sum "$OUT/gnb.conf")" ] || echo "WARNING: no change for: $1"; }
cp "$OUT/gnb.conf.orig" "$OUT/gnb.conf"
patch 's/mcc = [0-9]*;/mcc = 001;/'
patch 's/mnc = [0-9]*;/mnc = 01;/'
patch 's/tracking_area_code *= *[0-9]*;/tracking_area_code = 7;/'
patch 's/sd *= *0x[0-9a-fA-F]*;/sd = 0xffffff;/'
patch 's/ipv4 *= *"[0-9.]*";/ipv4 = "__AMF_IP__";/'
patch 's/GNB_IPV4_ADDRESS_FOR_NG_AMF *= *"[0-9./]*";/GNB_IPV4_ADDRESS_FOR_NG_AMF = "__POD_IP__";/'
patch 's/GNB_IPV4_ADDRESS_FOR_NGU *= *"[0-9./]*";/GNB_IPV4_ADDRESS_FOR_NGU = "10.100.3.10";/'
echo "wrote $OUT/gnb.conf -- review the diff against gnb.conf.orig before use"
