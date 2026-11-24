# How This System Works

## 1. What It Does

Two new LLM engines, Strata and Basalt, claim to run large models much faster than usual. Nobody independent has checked. This project checks, on your own GPU machine, and writes a short verdict.

An "engine" here means the program that loads a model onto the GPU and produces text from it. Ollama is the engine in use today, so it is the baseline.

**Input → What the system does → Final output**

- **Input:** a GPU node, plus a small settings file (`config.env`) that names the engine version and the model to test.
- **What it does:** runs the same model with the same questions through each engine, times everything, and grades the answers.
- **Final output:** `results/REPORT.md`, with three parts: does each engine run here, a table of measurements, and a one-paragraph recommendation.

## 2. The Whole Process in 60 Seconds

1. Check that the measuring tool itself works (a self-test).
2. Find a disk with enough room and decide which model size fits.
3. Look at the GPU. Rule out any engine the hardware cannot run.
4. Benchmark Ollama: download the model, time how long it takes to load, ask the fixed questions.
5. Benchmark Strata: build it inside a locked-down container, start it, ask the same questions.
6. Optionally benchmark vLLM, if you point the script at a running vLLM server.
7. Combine all results into one report and print a recommendation.

```
            config.env
                |
                v
  +---------------------------+
  | self-test the bench tool  |
  +---------------------------+
                |
                v
  +---------------------------+
  | pick disk + model size    |
  +---------------------------+
                |
                v
  +---------------------------+
  | inspect GPU, rule out     |---> status: "basalt cannot run"
  | engines that cannot run   |
  +---------------------------+
                |
      +---------+----------+----------------+
      v                    v                v
  +---------+        +-----------+     +------------+
  | Ollama  |        | Strata    |     | vLLM       |
  | (as is) |        | (sandbox) |     | (optional) |
  +---------+        +-----------+     +------------+
      |                    |                |
      +---- same 6 questions, same settings-+
                           |
                           v
              one result file per engine
              + one status line per engine
                           |
                           v
              +------------------------+
              | build REPORT.md        |
              | table + recommendation |
              +------------------------+
```

Everything is driven by one script, `run_all.sh`. It calls two helpers: `bench.py` asks the questions and measures, `report.py` writes the report.

## 3. The Major Jobs

### Job 1: Make Sure the Ruler Is Straight

**What happens.** Before touching any engine, the benchmark tool tests itself.

**Why.** A run takes hours and downloads over 100 GB. Finding a bug in the measuring code afterward would waste all of it.

**How.** The self-test checks three things:

- The synthetic patient notes come out identical every time, and the "known answers" really appear in the text.
- The grader gives the right grade for a perfect answer, a partly wrong answer, and an answer that is not valid JSON.
- The streaming client works. It starts a tiny fake server on the local machine, sends it a request, and checks that the reply text, token count and timing come back correctly.

**What comes out.** The line `selftest ok`, or the run stops.

### Job 2: Find Room for a Very Large Model

**What happens.** The script picks a folder to hold the model files and the build, and decides which model size to use.

**Why.** The model is about 111 GB. The root disk of a node is often too small, and guessing wrong means a failed download hours in.

**How.**

- It uses the folder from `config.env` if one is set. Otherwise it reuses the folder an earlier run chose. Otherwise it looks at every writable filesystem, skips RAM-backed ones, and takes the one with the most free space.
- It counts usable space as free space plus whatever earlier runs already stored there. That way a second run does not think the disk is full of its own downloads.
- It checks whether an Ollama server is already answering. If not, and the `ollama` program is installed, the script will start its own later. Its own Ollama stores a second copy of the model in the same folder, so more room is needed.

**The decision.**

| Usable space | Result |
|---|---|
| 160 GiB or more (275 if it runs its own Ollama) | Use the full 111 GB model |
| Less than that, but at least 110 GiB (198 with its own Ollama) | Switch both engines to a smaller 84 GB model pack |
| Less than 110 GiB | Stop with "This node cannot hold the test" |

**What comes out.** A data folder path (saved in `results/data_dir`) and a final choice of model. Both engines always get the same model, so the comparison stays fair.

### Job 3: Look at the Hardware and Rule Things Out Early

**What happens.** The script records the node's CPU type, GPU model, GPU count, memory, driver version and "compute capability" (a number NVIDIA uses to identify a GPU generation).

**Why.** Some engines only run on specific hardware. It is cheaper to say "cannot run" in one second than to fail after a two-hour build.

**How it decides.**

- **Basalt** needs an x86-64 CPU and a Blackwell-generation GPU (compute capability 12.0). If the node is anything else, Basalt is marked "cannot run" with the reason. If the node does qualify, Basalt is marked "not tested", because this script contains no Basalt run.
- **Strata's** check happens later, in Job 5: it needs NVIDIA driver 580.65 or newer.

**What comes out.** A hardware summary (`results/gpu.csv`) that goes straight into the report, and the first status line.

A "status line" is one row in `results/status.tsv`: engine name, result word (`runs`, `failed`, `cannot run`, `not run`, `skipped`, `not tested`), and a reason. Every job below adds one. If an engine gets several lines across reruns, the last one wins.

### Job 4: Measure the Baseline (Ollama)

**What happens.** Ollama runs the chosen model and answers the fixed questions.

**Why.** A speed number means nothing alone. "Strata does 9 tokens per second" only matters next to what Ollama does with the same model on the same machine.

**How.**

1. If no Ollama server is running and the program is installed, the script starts one, with its model store in the data folder.
2. If the server was already running, the script checks that the server's own model store has room for the download. If not, the engine is marked "not run" instead of filling someone's disk.
3. It asks Ollama to download the model. This does nothing if the model is already there.
4. It unloads the model, then loads it again with a stopwatch running. That is the **load time**.
5. It hands over to the question-and-measure tool (Job 6).
6. It unloads the model again, so the GPU is empty for the next engine.

**Optional extra.** If `OLLAMA_CURRENT_MODEL` is set, the same steps run a second time for the model you use day to day. That shows today's real speed next to the test model.

**What comes out.** `results/bench_ollama.json` and a status line.

### Job 5: Build and Run Strata Inside a Locked Box

**What happens.** Strata is downloaded, compiled, started and benchmarked, all inside a container.

**Why the box.** Strata is code from the internet that nobody here has reviewed. It gets built and run on a machine that may sit near sensitive data. A container (an isolated mini-system) limits what that code can reach.

**How the box is locked.**

- Strata never runs directly on the node. If no container tool is found (Docker, Podman or Apptainer), the script refuses rather than run it bare.
- It runs as your normal user, not as administrator, with all special privileges removed.
- It can see only two folders: a work folder and the model folder. Nothing else from the node is visible.
- The container's base system is pinned to one exact image fingerprint, so it cannot silently change.
- The Strata version is pinned to one exact 40-character commit ID. After checkout, the script verifies it really got that commit.
- Settings from `config.env` are checked against strict patterns before use. A setting containing odd characters stops the run. This matters because those values get written into a shell script.

**How the run goes.**

1. Check the driver is new enough. If not: "cannot run", and stop this job.
2. Check that Strata's port is free, so the script never benchmarks some other server by accident.
3. Inside the container: fetch the code, install build tools, compile the engine for this node's exact GPU generation, then let Strata's own setup download the model.
4. Start the Strata server in the background with a stopwatch running. Poll it every 3 seconds until it answers. That wait is Strata's **load time**. If the server dies or an hour passes, mark it "failed" with the last line of its log.
5. Hand over to the question-and-measure tool (Job 6).
6. Stop the server.

**Important design choice.** A failure here does not crash the whole run. It becomes a status line with a reason, and the script moves on. "Strata failed to build on this hardware" is a finding, and it still reaches the report.

**What comes out.** `results/bench_strata.json`, a status line, and two logs (`strata_setup.log`, `strata_server.log`).

**The vLLM side path.** vLLM is a third, mainstream engine. The script does not install it. If you set `VLLM_URL` and `VLLM_MODEL` to a server you already run, it gets the same questions. Otherwise it is marked "skipped".

### Job 6: Ask the Same Questions and Measure (the Core)

This job is the heart of the system. Jobs 4 and 5 both call it. It does not care which engine it talks to: all three speak the same OpenAI-style chat protocol, so one tool measures all of them the same way.

**The questions.** Six fixed prompts, all synthetic (no real patient data):

| Prompt | What it tests | Graded? |
|---|---|---|
| Summarize a 60-day hospital course | Long input, free-text output | No |
| Find 4 facts inside that same 60-day course | Long input, exact lookup | Yes |
| Pull 6 fields from a discharge note | Structured extraction | Yes |
| Pull 4 fields from a lab report full of OCR errors | Extraction from noisy text | Yes |
| Map 5 drugs to their doses | Structured extraction | Yes |
| Explain atrial fibrillation to a patient | Short input, free-text output | No |

The 60-day hospital course is generated by code with a fixed random seed, so it is the same roughly 2,000 words every time. Three facts are planted in it on purpose: a fever spike on day 23, an antibiotic change on day 12, and a specific lab value on day 17. Because the code planted them, it knows the right answers.

**Keeping it fair.**

- Every engine gets the same prompts, temperature 0 and a fixed seed, so answers are as repeatable as possible.
- The model's "thinking" mode is turned off for every engine. Each engine has its own switch for that, so the caller passes the right one in.
- Each request starts with a small tag such as `[request long_lookup-0]`. Some engines cache the start of a prompt and answer faster the second time. The tag defeats that. The tags are identical across engines, so nobody gets an advantage.
- One throwaway warm-up request is sent first, so the model is fully loaded before timing starts.

**What gets measured per request.** The answer arrives as a stream, piece by piece. That makes two timings possible:

- **Time to first token:** from sending the request to the first piece of the answer. This is mostly the time spent reading the prompt.
- **Generation speed (tokens per second):** tokens produced divided by the time from the first piece to the last. A token is a word fragment, the unit models produce. The count is `tokens − 1`, because the first token marks the start of the clock and so was not produced during it.

**How answers are graded.** For the four graded prompts, the tool finds the JSON object in the answer and compares it field by field with the known answers. Numbers match if they are equal as numbers (`"71.5"` equals `71.5`). Text matches ignoring case and outer spaces. The grade is the fraction of fields that are right. No valid JSON object means a grade of 0.

**Memory.** A background thread checks total GPU memory and RAM use once per second and keeps the highest value seen. Note that this is system-wide, not per program. Another job on the same node would inflate it.

**Rounds and failures.** The six prompts are repeated for several rounds (`ROUNDS`, default 3 in the tool). If one request fails, the error is recorded and the rest continue. The tool reports failure only if no request at all succeeded.

**What comes out.** One JSON file per engine, holding every request's timings, token counts, answer text and grade, plus load time and peak memory.

### Job 7: Turn the Numbers into a Verdict

**What happens.** The report tool reads every result file, the status lines and the hardware summary, and writes `REPORT.md`.

**Why.** The reader is a manager. They need a decision, with the evidence under it.

**How the table is built.** One row per engine:

- Generation speed: the median over all successful requests. A median is the middle value, so one slow outlier does not skew it.
- Time to first token, shown twice: median for short prompts and median for the two long ones. A single number would hide how each engine handles long input.
- Extraction accuracy: the average grade over all graded requests.
- Text similarity to baseline: how alike this engine's first-round answers are to Ollama's for the same prompts, from 0 to 1. A low value on the same model hints that the engine changed the output.
- Load time, peak VRAM, peak RAM, and count of failed requests.
- A warning, "(GPU not used?)", if peak GPU memory stayed under 4 GiB. That catches an engine quietly running on the CPU.

**How the recommendation is decided.** Rules are checked in this order:

1. Strata produced no numbers → "Stay on Ollama", with the reason from its status line.
2. Strata ran but there is no Ollama baseline → "No recommendation".
3. Strata is less than **1.5 times** as fast as Ollama → "Stay on Ollama". The bar is deliberately above 1.0: a small gain does not justify adopting an engine its authors never tested on this hardware.
4. Strata is fast enough, but its extraction accuracy is more than **0.05** below Ollama's → "Not recommended without a quality review". Fast and wrong is not a win.
5. Fast enough and accurate enough → "Worth a pilot on non-sensitive text", with advice to pin the commit.

**What comes out.** `results/REPORT.md`, also printed to the screen. The report step runs at the end of every invocation, even after a single step, so the report always reflects the latest results.

## 4. How the Pieces Work Together

The jobs talk to each other through plain files in the `results/` folder. Nothing is kept in memory between jobs.

| Produced by | File | Used by |
|---|---|---|
| Job 2 | `data_dir` | Later runs, to reuse the same folder |
| Job 3 | `gpu.csv` | The report's hardware section |
| Jobs 3, 4, 5 | `status.tsv` | The report's "Does it run here?" table and the recommendation's reason text |
| Job 6 | `bench_<engine>.json` | The report's measurement table and the recommendation |

Three consequences follow from this design:

- **Any step can be rerun alone.** `./run_all.sh strata` redoes only Strata. The Ollama results from yesterday are still on disk, so the report still compares both.
- **Earlier jobs feed later ones.** The model choice from Job 2 drives both engines. The GPU generation found in Job 3 tells Job 5 what to compile for. The load time measured in Jobs 4 and 5 is passed into Job 6, which copies it into the result file.
- **The engines never run at the same time.** Each one is unloaded or stopped before the next starts, so they do not compete for the GPU and the memory peaks stay meaningful.

## 5. Important External Pieces

- **Ollama.** The engine in use today, and the baseline. The script drives it over its local HTTP interface to download, load, unload and query models.
- **Strata.** The engine under test. Third-party, unreviewed code, fetched from GitHub at one pinned commit. Its own build and setup scripts are run, but only inside the container.
- **Basalt.** A fork of Strata. The script only checks whether the hardware could run it. It never downloads or runs Basalt.
- **vLLM.** An optional third engine. Only queried if you already run it.
- **Docker, Podman or Apptainer.** Container tools. Whichever one is installed becomes the locked box for Strata. Apptainer matters because shared research clusters often allow nothing else.
- **NVIDIA CUDA base image.** The starting system inside the container, holding the GPU compiler tools Strata's build needs.
- **`nvidia-smi`.** NVIDIA's command-line tool. It supplies the hardware facts and the GPU memory readings.
- **The model.** A large open model in GGUF format (a common file format for quantized models; "quantized" means compressed by storing numbers with less precision), downloaded once for each engine.
- **The OpenAI-style chat protocol.** The shared language all three engines speak. This is what lets one measuring tool serve them all.

The two Python tools use only the Python standard library. There is nothing to install.

## 6. End-to-End Example

The node and numbers below are made up to show the logic. The repository contains no real measurements.

**Setup.** An ARM64 node with one large NVIDIA GPU, compute capability 9.0, a recent driver, Docker installed, an Ollama server already running, and 300 GiB free on `/scratch`.

1. **Self-test.** `./run_all.sh` prints `selftest ok`.
2. **Disk.** `/scratch` has the most room. With 300 GiB usable and Ollama already running, the 160 GiB bar is met. The full 111 GB model is chosen. The data folder is `/scratch/<user>/strata-basalt-data`.
3. **Hardware.** The node is ARM64 with capability 9.0. Basalt needs x86-64 and 12.0. Status line: `basalt | cannot run | needs Linux x86-64 and a Blackwell GPU...`.
4. **Ollama.** The model downloads into Ollama's store. Unload, then a timed load: say 95 seconds. The six prompts run for three rounds, 18 requests. Suppose the median speed is 6.0 tokens per second and the average extraction grade is 0.95. Status: `ollama | runs`.
5. **Strata.** Driver check passes. Docker builds the container image. Inside it, Strata is cloned at the pinned commit, compiled for this GPU generation, and its setup downloads the model. The server starts and answers after, say, 140 seconds. The same 18 requests run. Suppose the median is 7.2 tokens per second and the grade is 0.94. Status: `strata | runs`.
6. **vLLM.** No URL was set. Status: `vllm | skipped`.
7. **Report.** The table gets two measured rows. Then the rules run:
   - Did Strata produce numbers? Yes.
   - Is there a baseline? Yes.
   - Speed ratio: 7.2 ÷ 6.0 = 1.20. That is below 1.5.
   - Verdict: "Strata generates 1.20x as fast as Ollama on the same model. That is below the 1.5x bar... Stay on Ollama."

The accuracy rule is never reached, because the speed rule already decided.

**A variation.** Had Strata reached 12.0 tokens per second (ratio 2.0) with a grade of 0.80, the speed rule would pass, but 0.80 is more than 0.05 below 0.95. The verdict would be "Not recommended without a quality review."

## 7. The Simplest Mental Model

It is a fair race with a referee.

It takes one large model and six fixed questions with known answers. It runs them through Ollama, then through Strata locked in a container, timing every answer and grading the ones that can be graded. It writes each engine's results to a file, then compares them using two simple rules: the new engine must be at least 1.5 times faster, and it must not be less accurate. Out comes a one-page report that says whether the new engine runs here, how it measured up, and whether to switch.
