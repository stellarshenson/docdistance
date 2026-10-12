#!/usr/bin/env bash
# Wait for the running re-execution lane to write its completion line, then re-execute the notebooks given.
# usage: reports/E13-campaign/reexec-after-lane.sh <lane count to wait for> <notebook>...
set -u
cd "$(dirname "$0")/../.." || exit 1
status=reports/E13-campaign/reexec-status.tsv
want=$1; shift
until [ "$(grep -c '^LANE-DONE' "$status")" -ge "$want" ]; do sleep 30; done
exec reports/E13-campaign/reexec.sh "$status" "$@"
