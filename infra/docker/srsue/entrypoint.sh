#!/bin/sh
# srsUE supervisor. The traffic generator's "reattach" episode writes the down time (seconds) to
# /run/srsue/reattach_delay_s and sends SIGINT to srsue: we wait that long, then start srsue again.
# This models a UE detach/attach cycle WITHOUT restarting the container (which would look like a
# pod failure in kube-state-metrics and blur the line with the pod_kill fault).
set -u
mkdir -p /run/srsue
resolve() { getent hosts "$1" | awk '{print $1; exit}'; }
while true; do
  GNB_IP=$(resolve "${GNB_HOST:-gnb-zmq.5g.svc.cluster.local}")
  sed -e "s/__GNB_IP__/${GNB_IP}/" /etc/srsran/ue.conf.tmpl > /tmp/ue.conf
  srsue /tmp/ue.conf &
  wait $!
  delay=$(cat /run/srsue/reattach_delay_s 2>/dev/null || echo 2)
  rm -f /run/srsue/reattach_delay_s
  echo "srsue exited; restarting in ${delay}s"
  sleep "${delay}"
done
