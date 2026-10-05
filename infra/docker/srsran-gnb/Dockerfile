# srsRAN Project gNB with the ZeroMQ virtual radio, pinned release.
# Build:  docker build -t testbed/srsran-gnb:25.10 infra/docker/srsran-gnb
ARG UBUNTU=24.04
FROM ubuntu:${UBUNTU} AS build
ARG SRSRAN_TAG=release_25_10
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git cmake make gcc g++ \
      pkg-config libfftw3-dev libmbedtls-dev libsctp-dev libyaml-cpp-dev libgtest-dev libzmq3-dev \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${SRSRAN_TAG} https://github.com/srsran/srsRAN_Project /src
# AUTO_DETECT_ISA=OFF: the image may be built on a machine with other CPU extensions than the
# one that runs it (illegal-instruction crashes otherwise). Costs a little speed.
RUN cd /src && mkdir build && cd build \
 && cmake .. -DENABLE_EXPORT=ON -DENABLE_ZEROMQ=ON -DAUTO_DETECT_ISA=OFF -DENABLE_UHD=OFF \
      -DCMAKE_BUILD_TYPE=Release \
 && make -j"$(nproc)" gnb && cp apps/gnb/gnb /usr/local/bin/gnb

FROM ubuntu:${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libfftw3-single3 libmbedcrypto7t64 \
      libsctp1 libyaml-cpp0.8 libzmq5 iproute2 iputils-ping procps \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/gnb /usr/local/bin/gnb
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
