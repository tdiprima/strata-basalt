# strata-basalt

A one-command, sandboxed benchmark that checks whether the Strata and Basalt LLM inference engines really beat Ollama on your own GPU node, and writes the answer as a short report.

## Speed claims nobody has checked

Large open models run slowly on self-hosted GPU nodes, and faster inference means more sensitive text processed without sending data off-site. Two new engines promise a large speed-up:

- **Strata** spreads a large mixture-of-experts model across GPU memory, system RAM and SSD.
- **Basalt**, a Strata fork, claims up to 2.6x Strata's speed.

Every one of those numbers is self-reported by the authors. No independent benchmark exists. Both projects are unreviewed third-party code, upstream has never tested the hardware in question, and a test run needs downloads of about 111 GB each. Deciding by reading READMEs is guesswork, and running unknown code on a node near sensitive data is a risk.

## A measured answer from one script

`./run_all.sh` runs the whole evaluation on the GPU node and ends with `results/REPORT.md`: does each engine run here, what was measured, and a recommendation.

**It compares like with like.** Every engine gets the same model, the same quantization, the same six prompts, temperature 0 and a fixed seed. Each request carries a fixed nonce so prefix caches cannot flatter one engine. The baseline is not a strawman: Ollama runs the same model at the same quantization as Strata.

**It measures quality, not only speed.** Four of the six prompts are structured-extraction tasks with known answers (a discharge note, a noisy OCR lab report, a medication list, and a lookup inside a 60-day hospital course). The answers are scored field by field, because aggressive quantization can be fast and wrong. All prompts are synthetic. No real patient data is used.

**It records what an operator needs.** Generation tokens per second, time to first token on short and long prompts, model load time, peak VRAM, peak RAM, extraction accuracy, text similarity to the baseline, and failed requests.

**It treats the engine as untrusted code.** Strata is cloned, built and run only inside a container, pinned to an exact 40-character commit, as a non-root user, with all capabilities dropped and only a work folder and a model folder mounted. The base image is pinned by digest. Config values are checked against allowlist patterns before they reach a shell.

**It needs no tuning.** The script picks the filesystem with the most free space, picks Docker, Podman or Apptainer, falls back to a smaller model pack when disk is short, and starts its own Ollama server if none is running. It also checks hardware requirements first, so an engine that cannot run on the node (Basalt needs x86-64 and a Blackwell GPU) is reported as such instead of failing halfway.

**It gives a decision, not a data dump.** The report recommends switching only if Strata is at least 1.5x as fast as Ollama and its extraction accuracy stays within 0.05 of the baseline. "Stay on Ollama" is a valid result.

The benchmark client and report generator use the Python standard library only: no dependencies to install or audit.

## What a run looks like

```sh
$ ./run_all.sh
selftest ok
data folder: /scratch/you/strata-basalt-data (… GiB usable); model: …
== basalt: cannot run (needs Linux x86-64 and a Blackwell GPU (compute capability 12.0); this node is …)
ollama long_summary r0: ttft … s, … tok/s, score None
ollama long_lookup r0: ttft … s, … tok/s, score …
...
== strata: runs (commit …, built for sm_… on …)
```

The run ends by printing `results/REPORT.md`, which has this layout (values elided here, since they depend on your node):

```markdown
# Strata and Basalt on this GPU node: result

## Node
CPU architecture, CUDA version, GPU model, VRAM, driver, compute capability

## Does it run here?
| Engine | Result | Detail |
|---|---|---|
| basalt | cannot run | needs Linux x86-64 and a Blackwell GPU ... |
| ollama | runs | ... |
| strata | runs | commit ..., built for sm_... |
| vllm | skipped | ... |

## Measured
| Engine | Model | Gen tok/s | TTFT short (s) | TTFT long prompt (s) | Load (s) | Peak VRAM (GiB) | Peak RAM (GiB) | Extraction accuracy | Text similarity to baseline | Failed requests |

## Recommendation
Strata generates N.NNx as fast as Ollama on the same model. ... Stay on Ollama.
```

The benchmark client also works alone against any OpenAI-compatible endpoint:

```sh
python3 bench.py --name strata --url http://127.0.0.1:8080/v1 --out results/bench_strata.json
```

## Usage

### Requirements

On the GPU node: `nvidia-smi`, `python3` (3.10 or newer), `curl`, and one of `docker`, `podman` or `apptainer`. At least 110 GiB of free disk. Strata's CUDA 13 build needs NVIDIA driver 580.65 or newer.

### Run it

1. Copy the folder to the node. The default target is the `lala` host of your `~/.ssh/config`.

   ```sh
   ./push.sh            # or: ./push.sh <host>
   ```

2. Log in and start the run inside `tmux`. It takes hours: two large downloads plus a build.

   ```sh
   ssh lala
   cd strata-basalt
   tmux new -s bench ./run_all.sh
   ```

   Detach with `Ctrl-b d`. Return with `tmux attach -t bench`.

3. Read the result.

   ```sh
   cat results/REPORT.md
   ```

### Run one step

```sh
./run_all.sh preflight   # check the node only
./run_all.sh ollama      # Ollama baseline
./run_all.sh strata      # build, load and benchmark Strata
./run_all.sh vllm        # optional third engine
./run_all.sh report      # rebuild REPORT.md from existing results
python3 bench.py --selftest
```

Downloads are kept between runs.

### Configuration

Nothing needs to be set. `config.env` holds overrides only:

| Variable | Purpose |
|---|---|
| `DATA_DIR` | Folder for models and the build, if the automatic choice is wrong |
| `RUNTIME` | Force `docker`, `podman` or `apptainer` |
| `STRATA_REPO`, `STRATA_COMMIT` | Upstream repository and the exact commit to test |
| `STRATA_FAMILY`, `STRATA_MODEL`, `STRATA_CONTEXT`, `STRATA_PORT` | Model pack, context size and server port for Strata |
| `OLLAMA_URL`, `OLLAMA_MODEL` | Ollama server and the same-model baseline |
| `OLLAMA_CURRENT_MODEL` | Optional: also benchmark the model you run today |
| `VLLM_URL`, `VLLM_MODEL` | Optional: an existing vLLM endpoint to include |
| `ROUNDS` | How many times each prompt is sent |

An optional bearer token for the endpoint is read from the `BENCH_API_KEY` environment variable.

### If something fails

- Build or download log: `results/strata_setup.log`
- Server log: `results/strata_server.log`
- "This node cannot hold the test": no filesystem has enough free space. Free some, or set `DATA_DIR` to a folder on a bigger disk.

### Safety and clean-up

Keep credentials and sensitive data out of the data folder the script prints at the start. Strata is untrusted code and can read that folder.

Delete the data folder when done:

```sh
rm -rf "$(cat results/data_dir)"
```

If the script used an Ollama server that was already running, the model is also in that server's own store. Remove it with `ollama rm <model>`.

<br>
