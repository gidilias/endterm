import json, os
d = r"C:\Users\ryuta\Desktop\AutoDAN_A3_merged"
for part in ("part1_200", "part2_400"):
    for rel in ("cases.json", r"runs\results.json", r"runs\source.json", r"runs\summary.json"):
        x = json.load(open(os.path.join(d, part, rel), encoding="utf-8"))
        print("\n==", part, rel, type(x).__name__, len(x))
        if isinstance(x, dict):
            print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in list(x.items())[:20]})
        else:
            f = x[0]
            print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in f.items()} if isinstance(f, dict) else type(f).__name__)
