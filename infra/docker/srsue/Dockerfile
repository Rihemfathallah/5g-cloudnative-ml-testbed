# srsUE (srsRAN 4G) with the ZeroMQ virtual radio, pinned release.
# Build:  docker build -t testbed/srsue:23.11 infra/docker/srsue
ARG UBUNTU=22.04
# srsRAN 4G 23.11 does not build cleanly with the GCC of Ubuntu 24.04; 22.04 is the tested base.
FROM ubuntu:${UBUNTU} AS build
ARG SRSRAN4G_TAG=release_23_11
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git cmake make gcc g++ \
      pkg-config libfftw3-dev libmbedtls-dev libboost-program-options-dev libconfig++-dev \
      libsctp-dev libzmq3-dev \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${SRSRAN4G_TAG} https://github.com/srsran/srsRAN_4G /src \
 && cd /src && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SRSENB=OFF -DENABLE_SRSEPC=OFF -DAUTO_DETECT_ISA=OFF \
 && make -j"$(nproc)" srsue && cp srsue/src/srsue /usr/local/bin/srsue

FROM ubuntu:${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libfftw3-single3 libmbedcrypto7 \
      libboost-program-options1.74.0 libconfig++9v5 libsctp1 libzmq5 iproute2 procps \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/srsue /usr/local/bin/srsue
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
