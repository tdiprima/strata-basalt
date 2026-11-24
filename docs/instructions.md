# Strata / Basalt test

Benchmarks Strata against Ollama on a GPU node and writes a short result report.

## Steps

1. On your own machine, copy this folder to the GPU node.

   ```sh
   ./push.sh
   ```

   It copies to the `lala` host of your `~/.ssh/config`. Another host: `./push.sh <host>`.

2. Log in to the node and enter the folder.

   ```sh
   ssh lala
   cd strata-basalt
   ```

3. Start it inside `tmux`, because it runs for hours (two downloads of about 111 GB, plus a build).

   ```sh
   tmux new -s bench
   ./run_all.sh
   ```

   Detach with `Ctrl-b d`. Come back with `tmux attach -t bench`.

4. Read the result.

   ```sh
   cat results/REPORT.md
   ```

5. Share `results/REPORT.md`.

## If something fails

- Strata build or download log: `results/strata_setup.log`
- Strata server log: `results/strata_server.log`
- Rerun one step: `./run_all.sh ollama`, `./run_all.sh strata`, or `./run_all.sh report`. Downloads are kept.
- "This node cannot hold the test": no filesystem has 110 GiB free. Free up space, or set `DATA_DIR` in `config.env` to a folder on a bigger disk.

## Rules

- Keep credentials and sensitive data out of the data folder the script prints. Strata is untrusted code and can read that folder.
- Needs on the node: `nvidia-smi`, `python3`, `curl`, and one of `docker`, `podman` or `apptainer`.

## Clean up

The script prints its data folder at the start and saves the path in `results/data_dir`. Delete that folder when done:

```sh
rm -rf "$(cat results/data_dir)"
```

If the script used an Ollama server that was already running, the model is also in Ollama's own store. Remove it with:

```sh
ollama rm hf.co/unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL
```

<br>
