import json, os, collections
d = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(d, "merged", "runs"); os.makedirs(out, exist_ok=True)

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

for name in ("results", "source"):
    rows = []
    for part in ("part1_200", "part2_400"):
        rows += load(os.path.join(d, part, "runs", f"{name}.json"))
    ids = [r["id"] for r in rows]
    print(f"\n{name}: {len(rows)} records, {len(set(ids))} unique ids")
    print("  by config:", dict(collections.Counter(r["config"] for r in rows)))
    print("  by seed:  ", dict(collections.Counter(r["seed"] for r in rows)))
    print("  by kind:  ", dict(collections.Counter(r["kind"] for r in rows)))
    with open(os.path.join(out, f"{name}_all.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
print("\nSaved to", out)
