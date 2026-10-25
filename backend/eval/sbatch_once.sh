#!/bin/bash
# sbatch that never double-submits.  Explorer's controller sometimes answers "Unexpected message received"
# while still accepting the job, so a blind retry makes a duplicate (two copies of a lane / render array ran
# into the same folders on 2026-10-23).  On an error, look for a job of this name submitted in the last few
# minutes before trying again.
#
#   eval/sbatch_once.sh -J <name> [sbatch args ...]      prints the job id (or FAILED)
name=""; prev=""
for x in "$@"; do [ "$prev" = "-J" ] && name="$x"; prev="$x"; done
[ -z "$name" ] && { echo "sbatch_once: needs -J <name>" >&2; exit 2; }
since=$(date -d "-1 min" +%Y-%m-%dT%H:%M:%S)
for try in 1 2 3 4; do
  if id=$(sbatch --parsable "$@" 2>/tmp/sbatch_once_$$); then echo "${id%%;*}"; exit 0; fi
  sleep 20
  id=$(sacct -u "$USER" -S "$since" -X -n -P -o jobid,jobname | awk -F'|' -v n="$name" '$2==n{print $1}' | sed 's/_.*//' | sort -u | tail -1)
  [ -n "$id" ] && { echo "$id"; exit 0; }            # it went through after all
  echo "sbatch_once: try $try failed: $(cat /tmp/sbatch_once_$$)" >&2
done
echo FAILED; exit 1
