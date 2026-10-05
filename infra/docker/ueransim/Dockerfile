# UERANSIM, pinned. Build: docker build -t testbed/ueransim:3.3.0 infra/docker/ueransim
FROM ubuntu:24.04 AS build
ARG UERANSIM_TAG=v3.3.0
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git make gcc g++ cmake \
      libsctp-dev lksctp-tools && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${UERANSIM_TAG} https://github.com/aligungr/UERANSIM /src && cd /src && make -j"$(nproc)"
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libsctp1 iproute2 iputils-ping iperf3 procps \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/nr-gnb /src/build/nr-ue /src/build/nr-cli /src/build/libdevbnd.so /usr/local/bin/
