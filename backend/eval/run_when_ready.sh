#!/usr/bin/env bash
# Wait for the Slakh download, verify it, extract it, then train.
#
# The verification is not ceremony.  A truncated 48 GB tarball extracts
# *partially* and silently — you get some tracks, no error, and a training run
# whose numbers mean nothing.  The download has already dropped once, so the
# size and MD5 are both checked against Zenodo's own record before anything
# reads the archive.
set -u

DIR=/d/slakh
TAR=$DIR/slakh2100_redux_16k.tar.gz
EXPECT_BYTES=48689473348
EXPECT_MD5=66a2301ed7b4d5f4f6d3383474e546c6
LOG=$DIR/pipeline.log
PROJ="C:/Users/HP/Documents/arrangement_transformer/Arrangement_Transformer/backend"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for download to reach $EXPECT_BYTES bytes"
STALL=0
LAST=0
while true; do
  SZ=$(stat -c %s "$TAR" 2>/dev/null || echo 0)
  [ "$SZ" -ge "$EXPECT_BYTES" ] && break
  if [ "$SZ" -eq "$LAST" ]; then
    STALL=$((STALL + 1))
  else
    STALL=0
  fi
  LAST=$SZ
  # curl has died before; if the file stops growing for 10 minutes, resume it
  # rather than waiting forever on a download that is no longer running.
  if [ "$STALL" -ge 20 ]; then
    say "stalled at $SZ bytes — resuming curl"
    ( cd "$DIR" && nohup curl -sL -C - -o "$TAR" \
      "https://zenodo.org/records/7708270/files/slakh2100_redux_16k.tar.gz?download=1" \
      >/dev/null 2>&1 & )
    STALL=0
  fi
  sleep 30
done
say "download complete: $(stat -c %s "$TAR") bytes"

say "verifying md5 (a few minutes for 48 GB)"
GOT=$(md5sum "$TAR" | cut -d' ' -f1)
if [ "$GOT" != "$EXPECT_MD5" ]; then
  say "MD5 MISMATCH: got $GOT, expected $EXPECT_MD5 — STOPPING"
  say "the archive is corrupt; delete it and re-download rather than extracting"
  exit 1
fi
say "md5 ok"

say "extracting"
tar -xzf "$TAR" -C "$DIR" || { say "extraction FAILED"; exit 1; }
# Ask the LOADER how many tracks it can see, not the shell.  Counting
# directories checks the wrong thing: 1,559 tracks extracted correctly and the
# loader still saw none, because the redux set is .flac and the code looked for
# mix.wav.  Training then ran to completion on the 20-track harness and said so
# in a single line nobody was watching.
cd "$PROJ" || exit 1
COUNTS=$(./venv/Scripts/python.exe -c "
from app.ml.transcriber.data import official_splits
tr, va, how = official_splits()
print(f'{len(tr)} {len(va)} {how}')" 2>/dev/null)
say "loader sees: $COUNTS"
TRAIN_N=$(echo "$COUNTS" | cut -d' ' -f1)
case "$COUNTS" in
  *official*) ;;
  *) say "loader did NOT find the official splits — STOPPING"; exit 1 ;;
esac
if [ "${TRAIN_N:-0}" -lt 100 ]; then
  say "too few tracks — STOPPING before training on a partial dataset"
  exit 1
fi

say "starting training (official splits, 40 epochs)"
./venv/Scripts/python.exe -u -m app.ml.transcriber.train --epochs 40 2>&1 \
  | tee -a "$LOG"
say "training finished"

say "running head-to-head against the NMF on the official validation split"
./venv/Scripts/python.exe -u eval/transcriber_vs_nmf.py --tracks 8 2>&1 \
  | tee -a "$LOG"
say "done"
