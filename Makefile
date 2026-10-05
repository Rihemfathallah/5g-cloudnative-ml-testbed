.PHONY: install test smoke handover-smoke render check-core check-sdn figures
install:        ; pip install -e ".[dev,bgp]"
test:           ; pytest
smoke:          ; tb smoke --fast
handover-smoke: ; tb handover all --fast -o results/smoke_ho
render:         ; python3 infra/open5gs/render.py k8s
check-core:     ; sudo -E python3 infra/open5gs/local_check.py
check-sdn:      ; sudo -E python3 infra/sdn/check_ring.py
figures:        ; figures/make_all.sh results/detection_main results/handover
