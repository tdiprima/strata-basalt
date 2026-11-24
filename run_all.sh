#!/usr/bin/env bash
# Test the Strata and Basalt engines against Ollama on this GPU node (task.md), and write results/REPORT.md.
#
#   ./run_all.sh              # everything: preflight, Ollama baseline, Strata, report
#   ./run_all.sh ollama       # one step: preflight | ollama | strata | vllm | report
#
# Nothing to set: it picks the disk, the container runtime and the model size, and downloads the model for both
# engines. config.env only holds overrides. Strata is untrusted code: it is cloned, built and run only inside
# a container, as your own non-root user, with nothing mounted except a work folder and the model folder.
# Do not put credentials or patient data under the data folder it prints.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
set -a; . ./config.env; set +a
RES=$HERE/results
IMAGE=strata-bench:local
PIDS=""
CT_NAME=strata-bench-$$
mkdir -p "$RES"
# Ctrl-C: the sandbox's own shell does not pass the signal on to the compile or download, so the container is removed by
# name, which kills everything in it. Long commands run through with_beat, where bash can act on the signal at once.
cleanup() {
  kill $PIDS 2>/dev/null || true
  case ${RUNTIME:-} in docker|podman) "$RUNTIME" rm -f "$CT_NAME" >/dev/null 2>&1 || true ;; esac
  echo "total time: $(( SECONDS / 3600 )) h $(( SECONDS % 3600 / 60 )) min"
}
trap cleanup EXIT
trap 'echo "run_all.sh: stopped" >&2; exit 130' INT TERM

die()  { echo "run_all.sh: $*" >&2; exit 1; }
# as root the sandbox drops every capability, so it cannot even read the work folder; and Strata must not get root
[ "$(id -u)" != 0 ] || die "do not run as root or with sudo; run as your own user"
note() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$RES/status.tsv"; echo "== $1: $2 ($3)"; }  # report.py reads the last line per engine
last_line() { tail -n 1 "$1" | tr '\t|' '  ' | cut -c1-200; }
# progress: step prints "[n/total time] what"; beat repeats the time and a log's newest line every minute of a long wait
N=0
exec 3>&1   # beat writes here, so it still reaches the screen when the command it watches is redirected into a log
step() { N=$(( N + 1 )); echo "[$N/$TOTAL $(date +%H:%M:%S)] $*"; }
stage() { local end=$(( N + $1 )); shift; "$@"; N=$end; }  # a stage that stops early still uses up its step numbers
beat() {
  local t0; t0=$(date +%s)
  while sleep 60; do
    echo "      $(( ($(date +%s) - t0) / 60 )) min: $(tail -c 2000 "$1" 2>/dev/null | tr '\r' '\n' | grep . | tail -n 1 | cut -c1-150)" >&3
  done
}
with_beat() {  # with_beat <log> <command...>
  local rc=0 b c
  beat "$1" & b=$!
  shift
  # in the background, then wait: bash runs the Ctrl-C trap during a wait, but not while a foreground command runs
  "$@" & c=$!
  PIDS="$PIDS $b $c"
  wait "$c" || rc=$?
  kill "$b" 2>/dev/null || true
  return "$rc"
}
# config values end up in shell and JSON text, so only plain characters are allowed
check() { [[ "$2" =~ $3 ]] || die "config.env: $1 has an unexpected value: $2"; }
check STRATA_REPO "$STRATA_REPO" '^https://github\.com/[A-Za-z0-9._/-]+$'
check STRATA_COMMIT "$STRATA_COMMIT" '^[0-9a-f]{40}$'
check STRATA_FAMILY "$STRATA_FAMILY" '^[A-Za-z0-9._-]+$'
check STRATA_MODEL "$STRATA_MODEL" '^[A-Za-z0-9._-]+$'
check STRATA_CONTEXT "$STRATA_CONTEXT" '^[0-9]+$'
check STRATA_PORT "$STRATA_PORT" '^[0-9]+$'
check ROUNDS "$ROUNDS" '^[0-9]+$'
check OLLAMA_MODEL "$OLLAMA_MODEL" '^[A-Za-z0-9._:/-]+$'
check OLLAMA_CURRENT_MODEL "${OLLAMA_CURRENT_MODEL:-x}" '^[A-Za-z0-9._:/-]+$'

# ------------------------------------------------------------------------------------------------ 0. disk and model size
ollama_up() { curl -fsS -m 10 "$OLLAMA_URL/api/version" >/dev/null 2>&1; }

pick_disk() {
  # The data folder: config.env, else the one an earlier run chose, else the writable filesystem with the most free
  # space (RAM-backed ones are left out). The root disk is only one of the node's filesystems.
  local mp d kb best="" best_kb=0
  if [ -z "${DATA_DIR:-}" ] && [ -s "$RES/data_dir" ]; then DATA_DIR=$(cat "$RES/data_dir"); fi
  if [ -z "${DATA_DIR:-}" ]; then
    for mp in "${SCRATCH:-}" "$HOME" $(df -Pk -x tmpfs -x devtmpfs -x squashfs -x overlay -x efivarfs | awk 'NR>1 {print $6}'); do
      for d in "$mp/$USER" "$mp"; do
        if [ -n "$mp" ] && [ -d "$d" ] && [ -w "$d" ]; then
          kb=$(df -Pk "$d" | awk 'NR==2 {print $4}')
          if [ "$kb" -gt "$best_kb" ]; then best=$d; best_kb=$kb; fi
          break
        fi
      done
    done
    [ -n "$best" ] || die "found no writable filesystem"
    DATA_DIR=$best/strata-basalt-data
  fi
  WORK=$DATA_DIR/strata-work
  mkdir -p "$WORK/home" "$DATA_DIR/models"
  echo "$DATA_DIR" > "$RES/data_dir"

  # No Ollama server answering: one is started from the ollama binary later, with its model store in the data folder.
  OWN_OLLAMA=0
  if ! ollama_up && command -v ollama >/dev/null; then OWN_OLLAMA=1; fi
  # Space that counts: free now plus what earlier runs already put in the data folder.
  local have=$(( ( $(df -Pk "$DATA_DIR" | awk 'NR==2 {print $4}') + $(du -sk "$DATA_DIR" | awk '{print $1}') ) / 1048576 ))
  MODEL_GB=111
  if [ "$have" -lt $(( 160 + OWN_OLLAMA * 115 )) ]; then
    # the smaller pack upstream ran on arm64 (84 GB); Ollama gets the same file set
    STRATA_FAMILY=qwen; STRATA_MODEL=IQ3_S; MODEL_GB=84
    OLLAMA_MODEL=hf.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_S
    if [ "$have" -lt $(( 110 + OWN_OLLAMA * 88 )) ]; then
      df -h >&2
      die "the roomiest writable filesystem ($DATA_DIR) has $have GiB; the smallest usable model needs $(( 110 + OWN_OLLAMA * 88 )) GiB. This node cannot hold the test."
    fi
  fi
  echo "data folder: $DATA_DIR ($have GiB usable); model: $STRATA_FAMILY $STRATA_MODEL"
}

# ------------------------------------------------------------------------------------------------ 1. the node
preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: run this on the GPU node"
  nvidia-smi > "$RES/nvidia-smi.txt"
  CPU_ARCH=$(uname -m)
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  VRAM_GIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {printf "%d", s/1024}')
  { echo "CPU architecture: $CPU_ARCH"
    grep -o 'CUDA Version: [0-9.]*' "$RES/nvidia-smi.txt" | head -n 1
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
  } > "$RES/gpu.csv"
  cat "$RES/gpu.csv"
  # Basalt README and docs/platforms.md (commit db8c8948): Linux x86-64 with AVX2, NVIDIA Blackwell (sm_120), CUDA 13.
  if [ "$CPU_ARCH" = x86_64 ] && [ "$CAP" = 12.0 ]; then
    note basalt "not tested" "this node meets Basalt's requirements, but this script has no Basalt run"
  else
    note basalt "cannot run" "needs Linux x86-64 and a Blackwell GPU (compute capability 12.0); this node is $CPU_ARCH with capability $CAP"
  fi
}

# ------------------------------------------------------------------------------------------------ 2. Ollama
ollama_api() { curl -sS --fail-with-body -m "${3:-60}" "$OLLAMA_URL/api/$1" -d "$2"; }  # body kept: it holds Ollama's reason

ollama_bench() {  # ollama_bench <label> <model>
  local name=$1 model=$2 t0 load store free
  ollama_up || { note "$name" "not run" "no Ollama server at $OLLAMA_URL and no ollama binary to start one"; return 0; }
  curl -fsS -m 10 "$OLLAMA_URL/api/version" > "$RES/ollama_version.json" || true
  # a running server keeps models in its own store: check that the download fits there, unless the model is already in it
  if [ "$OWN_OLLAMA" = 0 ] && ! curl -fsS -m 10 "$OLLAMA_URL/api/tags" | grep -qF "\"$model\""; then
    for store in "${OLLAMA_MODELS:-}" "$HOME/.ollama/models" /usr/share/ollama/.ollama/models; do
      if [ -n "$store" ] && [ -d "$store" ]; then
        free=$(df -Pk "$store" | awk 'NR==2 {printf "%d", $4/1048576}')
        [ "$free" -ge $(( MODEL_GB + 5 )) ] \
          || { note "$name" "not run" "Ollama's model store $store has $free GiB free; $model needs $MODEL_GB GiB"; return 0; }
        break
      fi
    done
  fi
  step "$name: download $model, $MODEL_GB GB (instant when already there; no detail until it ends)"
  with_beat /dev/null ollama_api pull "{\"model\":\"$model\",\"stream\":false}" 86400 > "$RES/ollama_pull_$name.json" \
    || { note "$name" "failed" "Ollama could not pull $model: $(last_line "$RES/ollama_pull_$name.json")"; return 0; }
  ollama_api generate "{\"model\":\"$model\",\"keep_alive\":0}" >/dev/null || true   # unload, so the load is timed
  step "$name: load the model onto the GPU"
  t0=$(date +%s)
  with_beat /dev/null ollama_api generate "{\"model\":\"$model\"}" 3600 >/dev/null \
    || { note "$name" "failed" "Ollama could not load $model"; return 0; }
  load=$(( $(date +%s) - t0 ))
  step "$name: benchmark"
  if python3 bench.py --name "$name" --url "$OLLAMA_URL/v1" --model "$model" --rounds "$ROUNDS" --load-s "$load" \
       --extra '{"reasoning_effort":"none"}' --out "$RES/bench_$name.json"; then
    note "$name" "runs" "$model"
  else
    note "$name" "failed" "no request succeeded; see the output above"
  fi
  curl -fsS -m 10 "$OLLAMA_URL/api/ps" > "$RES/ollama_ps_$name.json" || true          # size_vram shows the GPU share
  ollama_api generate "{\"model\":\"$model\",\"keep_alive\":0}" >/dev/null || true   # free the GPU for the next engine
}

ollama_step() {
  if [ "$OWN_OLLAMA" = 1 ] && ! ollama_up; then
    local host=${OLLAMA_URL#*://} i
    OLLAMA_MODELS="$DATA_DIR/ollama" OLLAMA_HOST="$host" ollama serve > "$RES/ollama_serve.log" 2>&1 &
    PIDS="$PIDS $!"
    for i in $(seq 30); do ollama_up && break; sleep 1; done
  fi
  ollama_bench ollama "$OLLAMA_MODEL"
  [ -z "${OLLAMA_CURRENT_MODEL:-}" ] || ollama_bench ollama-current "$OLLAMA_CURRENT_MODEL"
}

# ------------------------------------------------------------------------------------------------ 3. Strata
ct() {  # run a command in the sandbox. CT_EXEC=exec replaces the calling (background) shell, so its pid is the container's
  case $RUNTIME in
    docker|podman)
      local gpu=(--gpus all) user=(--user "$(id -u):$(id -g)") lim=(--ulimit memlock=-1:-1)
      if [ "$RUNTIME" = podman ]; then
        gpu=(--device nvidia.com/gpu=all); user=(--userns=keep-id); lim=(--ulimit host)
      fi
      # host network: the server binds 127.0.0.1 only, and the build needs to download. No capabilities, no new privileges.
      ${CT_EXEC:-} "$RUNTIME" run --rm --init --name "$CT_NAME" "${gpu[@]}" "${user[@]}" "${lim[@]}" --network host --cap-drop ALL \
        --security-opt no-new-privileges --security-opt label=disable --shm-size 2g -e HOME=/work/home \
        -v "$WORK:/work" -v "$DATA_DIR/models:/data" -w /work "$IMAGE" "$@" ;;
    apptainer)
      ${CT_EXEC:-} apptainer exec --nv --containall --no-home --bind "$WORK:/work,$DATA_DIR/models:/data" --pwd /work \
        --env HOME=/work/home "$WORK/strata.sif" "$@" ;;
  esac
}

strata_step() {
  local sm run pid t0 load base="http://127.0.0.1:$STRATA_PORT/v1"
  # Strata's arm64 build (engine/spark/build_engine.sh) needs the CUDA 13 toolkit, whose minimum driver is 580.65.
  if [ "$(printf '%s\n' 580.65 "$DRIVER" | sort -V | head -n 1)" != 580.65 ]; then
    note strata "cannot run" "its arm64 CUDA 13 engine needs NVIDIA driver 580.65 or newer; this node has $DRIVER"
    return 0
  fi
  sm=${CAP//./}
  check "GPU compute capability" "$sm" '^[0-9]+$'
  ! curl -fsS -m 3 "$base/models" >/dev/null 2>&1 || die "port $STRATA_PORT already answers; set another STRATA_PORT"

  step "strata: build the sandbox image"
  RUNTIME=${RUNTIME:-$(for r in docker podman apptainer; do command -v "$r" >/dev/null && { echo "$r"; break; }; done)}
  case ${RUNTIME:-} in
    docker|podman) "$RUNTIME" build -t "$IMAGE" - < Dockerfile ;;
    apptainer)     [ -f "$WORK/strata.sif" ] || apptainer build --fakeroot "$WORK/strata.sif" strata.def ;;
    *) die "no container runtime found (docker, podman or apptainer); Strata is not run outside a container" ;;
  esac || { note strata "failed" "$RUNTIME could not build the sandbox image; see the output above"; return 0; }

  # Runs inside the container. Upstream tested its arm64 build on the DGX Spark (sm_121) only; ARCHS aims it at this GPU.
  cat > "$WORK/steps.sh" <<EOF
set -eu
cd /work
[ -d Strata/.git ] || git clone --quiet $STRATA_REPO Strata
cd Strata
git checkout --quiet --detach $STRATA_COMMIT
[ "\$(git rev-parse HEAD)" = $STRATA_COMMIT ]
python3 -m venv /work/venv-build
/work/venv-build/bin/pip install --quiet -r terminal/requirements.txt -r terminal/requirements-build.txt
if [ ! -x terminal/engine/strata ]; then
  PATH=/work/venv-build/bin:\$PATH PYTHON=/work/venv-build/bin/python ARCHS=$sm BUILD_VISION=0 \\
    OUT=/work/Strata/terminal/engine sh engine/spark/build_engine.sh
fi
terminal/engine/strata --version
./setup.sh --setup --family $STRATA_FAMILY --model $STRATA_MODEL --context $STRATA_CONTEXT --vision no \\
  --yes --no-start --no-browser --port $STRATA_PORT --data-dir /data
EOF
  step "strata: compile ${STRATA_COMMIT:0:8} for sm_$sm, then download $STRATA_MODEL, $MODEL_GB GB (log: results/strata_setup.log)"
  if ! with_beat "$RES/strata_setup.log" ct sh /work/steps.sh > "$RES/strata_setup.log" 2>&1; then
    note strata "failed" "build or setup failed at commit ${STRATA_COMMIT:0:8}: $(last_line "$RES/strata_setup.log")"
    return 0
  fi
  run=$(ls -t "$WORK"/Strata/terminal/run-*.sh 2>/dev/null | head -n 1 || true)
  [ -n "$run" ] || { note strata "failed" "setup wrote no run script; see results/strata_setup.log"; return 0; }

  step "strata: start the server and load the model (log: results/strata_server.log)"
  t0=$(date +%s)
  beat "$RES/strata_server.log" & local b=$!
  PIDS="$PIDS $b"
  CT_EXEC="exec" ct sh "/work/Strata/terminal/$(basename "$run")" > "$RES/strata_server.log" 2>&1 &
  pid=$!
  PIDS="$PIDS $pid"
  until curl -fsS -m 5 "$base/models" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt 3600 ]; then
      kill "$pid" "$b" 2>/dev/null || true
      note strata "failed" "server did not come up: $(last_line "$RES/strata_server.log")"
      return 0
    fi
    sleep 3
  done
  kill "$b" 2>/dev/null || true
  load=$(( $(date +%s) - t0 ))
  step "strata: benchmark"
  if python3 bench.py --name strata --url "$base" --rounds "$ROUNDS" --load-s "$load" \
       --extra '{"chat_template_kwargs":{"enable_thinking":false}}' --out "$RES/bench_strata.json"; then
    note strata "runs" "commit ${STRATA_COMMIT:0:8}, $STRATA_FAMILY $STRATA_MODEL, built for sm_$sm on $CPU_ARCH (not a platform upstream tests)"
  else
    note strata "failed" "the server started but no request succeeded; see results/strata_server.log"
  fi
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

# ------------------------------------------------------------------------------------------------ 4. vLLM (optional)
vllm_step() {
  if [ -z "${VLLM_URL:-}" ] || [ -z "${VLLM_MODEL:-}" ]; then
    note vllm "skipped" "its Hopper recipe is 8 GPUs with a 172.78 GiB FP8 checkpoint; this node has $GPUS GPU(s), $VRAM_GIB GiB VRAM"
    return 0
  fi
  check VLLM_MODEL "$VLLM_MODEL" '^[A-Za-z0-9._:/-]+$'
  step "vllm: benchmark"
  if python3 bench.py --name vllm --url "$VLLM_URL" --model "$VLLM_MODEL" --rounds "$ROUNDS" \
       --extra '{"chat_template_kwargs":{"enable_thinking":false}}' --out "$RES/bench_vllm.json"; then
    note vllm "runs" "$VLLM_MODEL"
  else
    note vllm "failed" "no request succeeded"
  fi
}

O=3; [ -z "${OLLAMA_CURRENT_MODEL:-}" ] || O=6   # steps in the Ollama stage: download, load, benchmark, per model
case ${1:-all} in
  preflight|report) TOTAL=2 ;;
  ollama) TOTAL=$(( O + 2 )) ;;
  strata) TOTAL=6 ;;
  vllm)   TOTAL=3 ;;
  all)    TOTAL=$(( O + 7 )) ;;
  *) die "unknown step: $1" ;;
esac
step "check this node: self-test, disk, GPU"
python3 bench.py --selftest
pick_disk
preflight
case ${1:-all} in
  ollama) stage "$O" ollama_step ;;
  strata) stage 4 strata_step ;;
  vllm)   stage 1 vllm_step ;;
  all)    stage "$O" ollama_step; stage 4 strata_step; stage 1 vllm_step ;;
esac
step "write results/REPORT.md"
python3 report.py "$RES"
