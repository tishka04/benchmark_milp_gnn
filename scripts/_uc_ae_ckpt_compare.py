import sys, os, glob
sys.path.insert(0, os.getcwd())
import torch
from src.uc_jepa import UC_FEATURE_NAMES

paths = sorted(glob.glob("outputs/uc_jepa/autoencoder/uc_ae_*.pt"))
for p in paths:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = ck.get("metrics", {})
    print(f"{p}: epoch={ck.get('epoch')}  "
          f"loss={m.get('loss', float('nan')):.4f}  "
          f"ham={m.get('hamming', float('nan')):.4f}  "
          f"F1={m.get('macro_f1', float('nan')):.4f}  "
          f"R={m.get('macro_recall', float('nan')):.3f}  "
          f"P={m.get('macro_precision', float('nan')):.3f}")
    pf = m.get("per_feature_f1")
    if pf:
        for n, v in zip(UC_FEATURE_NAMES, pf):
            print(f"    {n:<22} {v:.3f}")
