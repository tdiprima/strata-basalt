# Test the Strata and Basalt inference engines

**What he said.** Two engines run LLMs much faster on limited GPU. Basalt is a fork of Strata. He wants to know whether the claims are real, and whether they would run on "the hoppers" next to Ollama. Today a ~100B-parameter model runs there at 5 to 8 tokens/second. Faster means more clinical free text and OCR processed in-house. If it turns out not to fit, that is an acceptable answer.

**What I found in a quick search** (secondary sources, not verified against the repos):

- **Strata** is an MIT-licensed engine for large mixture-of-experts models that do not fit in VRAM. It spreads the model across GPU memory, system RAM, and SSD. The headline claim is a 125B model (Qwen3.8-Flash-Next) on a 12 GB card. Repo: <https://github.com/ro99/strata>. A second repo, `Niko1221/Strata`, also appears in results, so work out which one is the real upstream.
- **Basalt** is a Strata fork. Its author claims up to 2.6x Strata's speed. It supports only Qwen3.8-Flash-Next and only NVIDIA Blackwell GPUs (RTX 50-series class). Support for older NVIDIA cards was removed. Repo: `github.com/jesdga95/basalt`.
- All speed numbers are self-reported by the authors. I found no independent benchmark.

**Likely outcome, to confirm.** If "hoppers" means NVIDIA Hopper GPUs (H100), Basalt cannot run there at all. Strata's main trick is fitting a big model into a small card, so on big-VRAM nodes the gain may be small. That would still be a valid, useful result.

**How to do it.**

1. On a hopper node run `nvidia-smi`. Record GPU model, count, VRAM, driver, and CUDA version.
2. Read both READMEs for hardware and model support. This alone may rule out Basalt.
3. Treat both repos as untrusted code. Read the build and install scripts first. Run in a container, as non-root, with no credentials or patient data mounted. Pin the exact commit you test.
4. Get a baseline with what you run today. Same model, same quantization, same prompts.

   ```sh
   ollama run <model> --verbose   # prints prompt eval rate and eval rate (tokens/s)
   ```

5. Run the same prompts through Strata (and Basalt, if hardware allows). Use 5 to 10 fixed prompts that include a long clinical-style text and a structured-extraction prompt.
6. Record for each engine: tokens/second for generation, time to first token, load time, peak VRAM and RAM, and whether output quality is the same. Aggressive quantization can be fast and wrong.
7. Optional third comparison: vLLM publishes a recipe for the same model that covers Hopper cards (<https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next>). It is the fair "normal fast engine" to compare against.
8. Write boss a short result: runs on our hardware or not, measured numbers in one table, recommendation.

