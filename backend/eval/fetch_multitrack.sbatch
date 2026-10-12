#!/bin/bash
# Download (many small byte ranges) + verify + extract a dataset onto scratch.
# Zenodo serves ~1.5 MB/s per connection AND drops connections after a few
# hundred MB, so the file is fetched as CHUNK-sized ranges (default 256 MB, a
# few minutes each), JOBS at a time.  Each range is one strict request (-f: an
# HTTP error is a failure, never data) into a temp file kept only at EXACTLY
# the range's length; a dropped chunk is refetched whole.  md5 at the end.
#   sbatch -J fetch-x --export=ALL,URL=...,MD5=...,NAME=... eval/fetch_multitrack.sbatch
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/fetch_%x_%j.out

set -e
CHUNK=${CHUNK:-268435456}
JOBS=${JOBS:-24}
D=/scratch/basu.ang/datasets/$NAME
mkdir -p $D/parts && cd $D
F=$(basename "${URL%%\?*}")
SIZE=$(curl -sfIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
N=$(( (SIZE + CHUNK - 1) / CHUNK ))
echo "$(date +%T) $F: $SIZE bytes in $N chunks, $JOBS at a time"
export URL SIZE CHUNK
fetch() {
  i=$1; a=$(( i * CHUNK )); b=$(( a + CHUNK - 1 )); [ $b -ge $SIZE ] && b=$(( SIZE - 1 ))
  want=$(( b - a + 1 ))
  for try in $(seq 1 40); do
    [ "$(stat -c %s parts/$i 2>/dev/null || echo 0)" = "$want" ] && return 0
    if curl -sfL --max-time 1800 -r $a-$b -o parts/$i.tmp "$URL" \
       && [ "$(stat -c %s parts/$i.tmp)" = "$want" ]; then
      mv parts/$i.tmp parts/$i; return 0
    fi
    rm -f parts/$i.tmp; sleep $(( 5 + try ))
  done
  echo "chunk $i gave up"; return 1
}
export -f fetch
seq 0 $(( N - 1 )) | xargs -P $JOBS -I{} bash -c 'fetch {}'
for i in $(seq 0 $(( N - 1 ))); do [ -f parts/$i ] || { echo "chunk $i missing"; exit 1; }; done
echo "$(date +%T) chunks done; joining"
for i in $(seq 0 $(( N - 1 ))); do cat parts/$i; done > "$F" && rm -rf parts
echo "$MD5  $F" | md5sum -c -
echo "$(date +%T) extracting"
case "$F" in
  *.zip) unzip -q -o "$F" ;;
  *.tar.gz) tar xzf "$F" ;;
esac
rm -f "$F"
echo "$(date +%T) done: $(du -sh . | cut -f1) in $D"
