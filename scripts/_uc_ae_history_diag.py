import json, sys, os
sys.path.insert(0, os.getcwd())
from src.uc_jepa import UC_FEATURE_NAMES

h = json.load(open("outputs/uc_jepa/autoencoder/history.json"))
print(f"epochs in history: {len(h)}")

print("\nVal macro metrics by epoch:")
print(f"{'ep':>3} {'loss':>7} {'ham':>6} {'F1':>6} {'R':>6} {'P':>6}")
for r in h:
    v = r["val"]
    print(f"{r['epoch']:>3} {v['loss']:>7.3f} {v['hamming']:>6.3f} "
          f"{v.get('macro_f1', float('nan')):>6.3f} "
          f"{v.get('macro_recall', float('nan')):>6.3f} "
          f"{v.get('macro_precision', float('nan')):>6.3f}")

print("\nPer-feature F1 trajectory:")
print("       " + "  ".join(f"{n[:10]:>10}" for n in UC_FEATURE_NAMES))
for r in h:
    pf = r["val"].get("per_feature_f1", [float("nan")] * 8)
    print(f"ep{r['epoch']:02d}   " + "  ".join(f"{x:>10.3f}" for x in pf))
