# UE-side probe sidecar (and data-network iperf3 server). Build: docker build -t testbed/probe:1.0 infra/docker/probe
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iperf3 iputils-ping iproute2 procps \
 && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir prometheus_client==0.21.1
COPY probe.py /probe.py
CMD ["python3", "/probe.py"]
