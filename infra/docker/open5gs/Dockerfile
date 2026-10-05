# Open5GS 5G core, built from source at a pinned tag (the same build is validated by
# infra/open5gs/local_check.sh). Build:  docker build -t testbed/open5gs:2.7.7 infra/docker/open5gs
ARG UBUNTU=24.04
FROM ubuntu:${UBUNTU} AS build
ARG OPEN5GS_TAG=v2.7.7
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git build-essential \
      meson ninja-build flex bison pkg-config libsctp-dev libgnutls28-dev libgcrypt-dev libssl-dev \
      libmongoc-dev libbson-dev libyaml-dev libnghttp2-dev libmicrohttpd-dev libcurl4-gnutls-dev \
      libtins-dev libtalloc-dev libidn-dev cmake python3 \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${OPEN5GS_TAG} https://github.com/open5gs/open5gs /src \
 && cd /src && meson setup build --prefix=/opt/open5gs \
 && ninja -C build && ninja -C build install

FROM ubuntu:${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libsctp1 libgnutls30 \
      libgcrypt20 libssl3 libmongoc-1.0-0 libbson-1.0-0 libyaml-0-2 libnghttp2-14 libmicrohttpd12 \
      libcurl3t64-gnutls libtins4.5 libtalloc2 libidn12 iproute2 iptables iputils-ping curl procps \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/open5gs /opt/open5gs
ENV PATH=/opt/open5gs/bin:$PATH LD_LIBRARY_PATH=/opt/open5gs/lib/x86_64-linux-gnu
