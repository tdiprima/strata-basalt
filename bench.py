#!/usr/bin/env python3
"""Benchmark one OpenAI-compatible endpoint (Ollama, Strata, vLLM) with fixed prompts.

    python3 bench.py --name strata --url http://127.0.0.1:8080/v1 --out results/bench_strata.json
    python3 bench.py --selftest

Records per request: time to first token, generation tokens/s, token counts, the answer, and a score
for the extraction prompts (they have a known answer). Samples GPU memory and RAM once a second.
Every prompt is synthetic: no real patient data. Standard library only.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

API_KEY_ENV = "BENCH_API_KEY"  # optional bearer token, read from the environment only


# ---------------------------------------------------------------------------------------------- prompts
def ward_notes(days=60):
    """A long synthetic hospital course (about 2,000 words, a few thousand tokens) and the facts a model must find in it."""
    rng = random.Random(7)
    pick = lambda items: items[int(rng.random() * len(items))]
    fever_day, switch_day, lab_day = 23, 12, 17
    facts, lines = {}, []
    for d in range(1, days + 1):
        temp = 39.4 if d == fever_day else round(36.4 + rng.random() * 1.6, 1)
        creat = round(0.80 + rng.random() * 0.70, 2)
        if d == lab_day:
            facts["creatinine_day_17"] = creat
        lines.append(
            f"Hospital day {d}. Temp {temp} C, HR {int(60 + rng.random() * 50)}, "
            f"BP {int(100 + rng.random() * 50)}/{int(55 + rng.random() * 35)}, SpO2 {int(92 + rng.random() * 8)}% on "
            f"{pick(['room air', '2 L nasal cannula', '4 L nasal cannula'])}. Creatinine {creat} mg/dL, "
            f"WBC {round(4 + rng.random() * 14, 1)}. "
            f"{pick(['Slept poorly overnight.', 'Ambulated in the hallway with physiotherapy.', 'Reports mild nausea after meals.', 'Appetite improving.', 'Family at bedside, questions answered.', 'Complains of fatigue.'])} "
            f"{pick(['Lungs with bibasilar crackles.', 'Abdomen soft, non-tender.', 'Wound edges clean and dry.', 'No peripheral edema.', 'Trace ankle edema.'])} "
            + ("Antibiotic changed from ceftriaxone to meropenem after culture results. " if d == switch_day else "")
            + f"Plan: {pick(['continue current management', 'repeat labs in the morning', 'encourage oral intake', 'wean oxygen as tolerated', 'continue physiotherapy'])}."
        )
    facts.update(max_temp_day=fever_day, max_temp_c=39.4, antibiotic_changed_day=switch_day)
    return "\n".join(lines), facts


def prompts():
    """The fixed prompt set: id, user text, max_tokens, and the expected JSON fields (None = not scored)."""
    notes, facts = ward_notes()
    js = "Answer with one JSON object only, no other text. "
    return [
        ("long_summary", "Synthetic progress notes follow.\n\n" + notes +
         "\n\nSummarize the hospital course in at most 150 words.", 300, None),
        ("long_lookup", "Synthetic progress notes follow.\n\n" + notes + "\n\n" + js +
         "Keys: max_temp_day (integer day with the highest temperature), max_temp_c (number), "
         "creatinine_day_17 (number), antibiotic_changed_day (integer).", 120, facts),
        ("extract_discharge", js + "Keys: age (integer), sex (\"M\" or \"F\"), principal_diagnosis (string), "
         "allergy (string), follow_up_days (integer), discharge_weight_kg (number).\n\n"
         "DISCHARGE NOTE (synthetic). A 67-year-old woman was admitted with shortness of breath and treated for "
         "community-acquired pneumonia. She has a documented allergy to penicillin. Past history includes type 2 "
         "diabetes and hypertension. She improved on levofloxacin and was discharged at a weight of 71.5 kg with "
         "instructions to see her primary care physician in 14 days.", 150,
         {"age": 67, "sex": "F", "principal_diagnosis": "community-acquired pneumonia", "allergy": "penicillin",
          "follow_up_days": 14, "discharge_weight_kg": 71.5}),
        ("extract_ocr", "The text below is OCR output with recognition errors. " + js +
         "Keys: mrn (string of digits), dob (YYYY-MM-DD), hemoglobin_g_dl (number), potassium_mmol_l (number).\n\n"
         "LAB0RATORY REP0RT (synthet1c)\nPat1ent MRN: 0O48213 7  ->  00482137\nD0B: 1958-O3-2l (March 21, 1958)\n"
         "Hemog1obin ...... 9.8 g/dL  L\nPotass1um ....... 5.6 mmo1/L  H\nSod1um .......... 138 mmol/L", 120,
         {"mrn": "00482137", "dob": "1958-03-21", "hemoglobin_g_dl": 9.8, "potassium_mmol_l": 5.6}),
        ("extract_meds", js + "Map each drug name (lowercase generic name) to its dose in mg per administration.\n\n"
         "Medications on discharge (synthetic): metformin 500 mg twice daily with meals; lisinopril 10 mg once "
         "daily; atorvastatin 40 mg at bedtime; furosemide 20 mg every morning; apixaban 5 mg twice daily.", 150,
         {"metformin": 500, "lisinopril": 10, "atorvastatin": 40, "furosemide": 20, "apixaban": 5}),
        ("explain", "Explain to a patient, in about 200 words and plain language, what atrial fibrillation is and "
         "why a blood thinner may be prescribed.", 350, None),
    ]


def score(text, expect):
    """Fraction of expected fields the answer's JSON object gets right (0.0 when there is no valid JSON object)."""
    try:
        got = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return 0.0
    if not isinstance(got, dict):
        return 0.0
    got = {str(k).strip().lower(): v for k, v in got.items()}

    def same(a, b):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return str(a).strip().lower() == str(b).strip().lower()
    return sum(same(got.get(k), v) for k, v in expect.items()) / len(expect)


# ---------------------------------------------------------------------------------------------- client
def chat(base, model, user, max_tokens, extra):
    """One streamed chat completion. Returns ttft_s, gen_tps, wall_s, token counts and the answer text."""
    body = {"model": model, "messages": [{"role": "user", "content": user}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 1, "stream": True, "stream_options": {"include_usage": True}, **extra}
    headers = {"Content-Type": "application/json"}
    if os.environ.get(API_KEY_ENV):
        headers["Authorization"] = "Bearer " + os.environ[API_KEY_ENV]
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0, first, text, usage = time.perf_counter(), None, [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            obj = json.loads(line[5:])
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or ""
                if first is None and (piece or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.perf_counter()
                text.append(piece)
            usage = obj.get("usage") or usage
    end = time.perf_counter()
    n = usage.get("completion_tokens")
    gen = end - first if first else None
    return {"ttft_s": round(first - t0, 3) if first else None, "wall_s": round(end - t0, 3),
            # the first token lands at `first`, so n-1 tokens were produced during `gen`
            "gen_tps": round((n - 1) / gen, 2) if n and n > 1 and gen else None,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n, "text": "".join(text)}


def memory_now():
    """(GPU memory used over all GPUs in MiB or None, RAM used in GiB or None), system-wide."""
    vram = ram = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        vram = sum(int(x) for x in out.split()) if out.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        info = dict(l.split(":") for l in open("/proc/meminfo"))
        ram = round((int(info["MemTotal"].split()[0]) - int(info["MemAvailable"].split()[0])) / 2**20, 1)
    except (OSError, KeyError, ValueError):
        pass
    return vram, ram


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", help="engine label in the report")
    ap.add_argument("--url", help="OpenAI-compatible base URL, for example http://127.0.0.1:8080/v1")
    ap.add_argument("--model", help="model name (default: the first one GET /models lists)")
    ap.add_argument("--out", help="result file (JSON)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--load-s", type=float, help="model load time measured by the caller, copied into the result")
    ap.add_argument("--extra", default="{}", help="extra request fields as JSON (how this engine turns thinking off)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.name and a.url and a.out):
        ap.error("--name, --url and --out are required")
    if not a.url.startswith(("http://", "https://")):
        ap.error("--url must be http or https")
    base, extra = a.url.rstrip("/"), json.loads(a.extra)
    model = a.model or json.load(urllib.request.urlopen(base + "/models", timeout=30))["data"][0]["id"]

    start, stop = memory_now(), threading.Event()
    peak = {"vram": start[0], "ram": start[1]}

    def sample():
        while not stop.wait(1):
            for key, v in zip(("vram", "ram"), memory_now()):
                if v is not None:
                    peak[key] = max(peak[key] or 0, v)
    threading.Thread(target=sample, daemon=True).start()

    chat(base, model, "Reply with the word ready.", 8, extra)  # warm-up: the model is loaded before timing starts
    runs = []
    for rnd in range(a.rounds):
        for pid, user, max_tokens, expect in prompts():
            # a fixed nonce per (prompt, round): defeats prefix caches, and is the same for every engine
            row = {"prompt": pid, "round": rnd}
            try:
                row.update(chat(base, model, f"[request {pid}-{rnd}]\n{user}", max_tokens, extra))
                row["score"] = score(row["text"], expect) if expect else None
            except Exception as e:  # one failed request must not lose the others
                row["error"] = f"{type(e).__name__}: {e}"
            runs.append(row)
            print(f"{a.name} {pid} r{rnd}: " + (row.get("error") or
                  f"ttft {row['ttft_s']} s, {row['gen_tps']} tok/s, score {row['score']}"), flush=True)
    stop.set()
    result = {"name": a.name, "url": base, "model": model, "load_s": a.load_s, "rounds": a.rounds,
              "vram_start_mib": start[0], "vram_peak_mib": peak["vram"], "ram_start_gib": start[1],
              "ram_peak_gib": peak["ram"], "runs": runs}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    good = [r["gen_tps"] for r in runs if r.get("gen_tps")]
    print(f"{a.name}: median {round(statistics.median(good), 1) if good else 'n/a'} tok/s over {len(good)}/{len(runs)} requests")
    return 0 if good else 1


def selftest():
    """Checks the prompt set, the scorer and the streaming client (against a local fake server)."""
    import http.server
    notes, facts = ward_notes()
    assert ward_notes() == (notes, facts) and len(notes.split()) > 1500
    assert f"Hospital day 17." in notes and f"Creatinine {facts['creatinine_day_17']} mg/dL" in notes
    assert max(float(l.split("Temp ")[1].split()[0]) for l in notes.splitlines()) == 39.4
    assert len(prompts()) == 6 and len({p[0] for p in prompts()}) == 6
    exp = {"age": 67, "sex": "F", "w": 71.5}
    assert score('```json\n{"age": 67, "sex": "f", "w": "71.5"}\n```', exp) == 1.0
    assert score('{"Age": 66, "sex": "F"}', exp) == 1 / 3
    assert score("no json here", exp) == 0.0 and score("[1]{", exp) == 0.0

    class Fake(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert req["stream"] and req["temperature"] == 0 and req["x"] == 1
            self.send_response(200)
            self.end_headers()
            for piece in ('{"a"', ": 1}"):
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}\n\ndata: [DONE]\n\n')

    srv = http.server.HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    r = chat(f"http://127.0.0.1:{srv.server_port}", "m", "hi", 8, {"x": 1})
    srv.shutdown()
    assert r["text"] == '{"a": 1}' and r["completion_tokens"] == 3 and r["ttft_s"] is not None and r["gen_tps"] > 0, r
    assert score(r["text"], {"a": 1}) == 1.0
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
