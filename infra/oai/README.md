# OpenAirInterface profile (optional, time-boxed: one week)

Purpose: run ONE collection run with a second RAN stack (OAI gNB + nrUE in RF-simulator mode)
on the same core and SDN transport, and apply detectors trained on srsRAN runs to it
(`tb windows <srsran runs> --test-only <oai run>`).

OAI configuration files are long libconfig files maintained upstream; copying them here from
memory would be error-prone. `fetch-configs.sh` downloads the RF-simulator SA config of the
pinned OAI tag (infra/VERSIONS.md) and patches only the fields this testbed needs:
PLMN 001/01, TAC 7, SST 1, the AMF address, and the N2/N3 interface addresses
(N3 = 10.100.3.10 on interface n3, exactly like the srsRAN gNB).

Status: NOT validated. If it does not register a UE within the time box, remove
OpenAirInterface from the CV tools list (see the architecture doc, section 1).
