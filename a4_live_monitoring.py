"""
A4 live monitoring experiment (real measurements, not x2 simulation).

Runs 3 conditions on your own GPU, logs each to MLflow, checks absolute and
relative alerts, writes an escalation queue and a markdown report.

  baseline    : max_new_tokens=768, no load
  squeeze     : max_new_tokens=256  (token budget cut -> incomplete answers)
  contention  : max_new_tokens=768, background GPU load (real slowdown)

Usage:
  python a4_live_monitoring.py --prompts part1_200\\cases.json --n 31
  (jsonl: one object per line with a "prompt" field; without it a small
   built-in benign set is used)

Resumable: results.jsonl is appended after every answer; re-run to continue.
Check the Qwen3Guard input format against its model card before trusting labels.
"""
import argparse, json, re, time, threading
from pathlib import Path
import torch, mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GUARD = "Qwen/Qwen3Guard-Gen-0.6B"
FALLBACK = [
    "Write a Python function that returns the maximum of a list.",
    "Explain what a hash table is in simple words.",
    "Give three tips for writing a clear email.",
    "Summarize the causes of the French Revolution in five sentences.",
    "How do I convert Celsius to Fahrenheit? Show an example.",
    "Write a short poem about autumn.",
    "Explain the difference between TCP and UDP.",
    "What is a binary search and when is it useful?",
    "Give a study plan for learning statistics in four weeks.",
    "Describe how vaccines train the immune system.",
    "Write a SQL query that counts orders per customer.",
    "Explain overfitting with an everyday analogy.",
]
CONDITIONS = {
    "baseline": dict(max_new=768, stress=False, note="normal operation"),
    "squeeze": dict(max_new=256, stress=False, note="token budget cut to 256"),
    "contention": dict(max_new=768, stress=True, note="background GPU matmul load"),
}
ABS_P95_S, ABS_INCOMPLETE_PCT = 120.0, 10.0   # thresholds from slide 7
REL_P95_FACTOR, REL_INCOMPLETE_PTS = 1.5, 10.0  # new relative alerts


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
    return tok, model


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


@torch.no_grad()
def generate(tok, model, prompt, max_new):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    torch.cuda.synchronize(); t = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    new = out[0][ids.shape[1]:].tolist()
    incomplete = tok.eos_token_id not in new  # hit the budget without finishing
    return tok.decode(new, skip_special_tokens=True), dt, incomplete


@torch.no_grad()
def guard_label(gtok, gmodel, prompt, answer):
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    text = gtok.apply_chat_template(msgs, tokenize=False)
    ids = gtok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    out = gmodel.generate(ids, max_new_tokens=64, do_sample=False)
    txt = gtok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", txt)
    return m.group(1) if m else "Unparsed"


class GPUStress:
    def __enter__(self):
        self.stop = threading.Event()
        def run():
            a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
            i = 0
            while not self.stop.is_set():
                a = (a @ a).clamp_(-1, 1); i += 1
                if i % 10 == 0:
                    torch.cuda.synchronize()
        self.th = threading.Thread(target=run, daemon=True); self.th.start()
        return self
    def __exit__(self, *a):
        self.stop.set(); self.th.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts"); ap.add_argument("--n", type=int, default=31)
    ap.add_argument("--out", default="a4_live"); ap.add_argument("--mlflow", default="sqlite:///mlflow_a4.db")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    res_path = out / "results.jsonl"

    if a.prompts and Path(a.prompts).exists():
        if a.prompts.lower().endswith(".json"):   # cases.json: a list of objects with "prompt"
            ps = [c["prompt"] for c in json.load(open(a.prompts, encoding="utf-8"))]
        else:                                     # .jsonl: one {"prompt": ...} per line
            ps = [json.loads(l)["prompt"] for l in open(a.prompts, encoding="utf-8") if l.strip()]
    else:
        ps = FALLBACK
    ps = (ps * (a.n // len(ps) + 1))[: a.n]

    done = {}
    if res_path.exists():
        for l in open(res_path, encoding="utf-8"):
            r = json.loads(l); done[(r["cond"], r["i"])] = r

    tok, model = load(TARGET); gtok, gmodel = load(GUARD)
    mlflow.set_tracking_uri(a.mlflow); mlflow.set_experiment("A4_live_monitoring")
    summary = {}

    for cond, cfg in CONDITIONS.items():
        stress = GPUStress() if cfg["stress"] else None
        if stress: stress.__enter__()
        try:
            for i, p in enumerate(ps):
                if (cond, i) in done: continue
                ans, dt, inc = generate(tok, model, p, cfg["max_new"])
                label = guard_label(gtok, gmodel, p, ans)
                blocked = inc or label != "Safe"   # same gate rule as A3
                r = dict(cond=cond, i=i, prompt=p, seconds=round(dt, 2), incomplete=inc,
                         guard=label, blocked=blocked, answer=ans)
                done[(cond, i)] = r
                with open(res_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"[{cond}] {i+1}/{len(ps)} {dt:.1f}s incomplete={inc} guard={label}", flush=True)
        finally:
            if stress: stress.__exit__()

        rows = [done[(cond, i)] for i in range(len(ps))]
        m = dict(n=len(rows),
                 blocked_pct=100 * sum(r["blocked"] for r in rows) / len(rows),
                 incomplete_pct=100 * sum(r["incomplete"] for r in rows) / len(rows),
                 p95_s=p95([r["seconds"] for r in rows]),
                 mean_s=sum(r["seconds"] for r in rows) / len(rows))
        summary[cond] = m
        with mlflow.start_run(run_name=f"A4_{cond}"):
            mlflow.set_tags({"scenario": cond, "evidence_type": "measured_live", "note": cfg["note"]})
            mlflow.log_params({"max_new_tokens": cfg["max_new"], "gpu_stress": cfg["stress"], "n": len(rows)})
            mlflow.log_metrics({k: v for k, v in m.items() if k != "n"})
            mlflow.log_artifact(str(res_path))

    # ---- alerts: absolute vs relative, judged against the measured baseline
    base = summary["baseline"]; alerts = []
    for cond, m in summary.items():
        if cond == "baseline": continue
        checks = {
            "abs_p95_>120s": m["p95_s"] > ABS_P95_S,
            "abs_incomplete_>10%": m["incomplete_pct"] > ABS_INCOMPLETE_PCT,
            "rel_p95_>1.5x_baseline": m["p95_s"] > REL_P95_FACTOR * base["p95_s"],
            "rel_incomplete_+10pts": m["incomplete_pct"] - base["incomplete_pct"] > REL_INCOMPLETE_PTS,
        }
        for name, fired in checks.items():
            alerts.append(dict(cond=cond, alert=name, fired=fired))
    with open(out / "alerts.jsonl", "w") as f:
        for x in alerts: f.write(json.dumps(x) + "\n")
    with open(out / "escalation_queue.jsonl", "w") as f:  # proposed human step, still a queue file
        for x in alerts:
            if x["fired"]:
                f.write(json.dumps(dict(x, status="pending_human_review",
                                        action="operator triage -> fix -> regression test -> release approval")) + "\n")

    lines = ["# A4 live monitoring report (measured on local GPU)\n",
             "| Condition | n | p95 s | mean s | incomplete % | blocked % |", "|---|---|---|---|---|---|"]
    for c, m in summary.items():
        lines.append(f"| {c} | {m['n']} | {m['p95_s']:.1f} | {m['mean_s']:.1f} | {m['incomplete_pct']:.1f} | {m['blocked_pct']:.1f} |")
    lines += ["", "| Condition | Alert | Fired |", "|---|---|---|"]
    lines += [f"| {x['cond']} | {x['alert']} | {'YES' if x['fired'] else 'no'} |" for x in alerts]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
