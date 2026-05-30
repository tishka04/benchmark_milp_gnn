"""Sweep decision threshold per feature to see the operating-point ceiling."""
import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.uc_jepa.data.uc_label_dataset import UCLabelDataset, uc_collate
from src.uc_jepa.models.uc_autoencoder import UCAutoencoder
from src.uc_jepa import UC_FEATURE_NAMES

ck = torch.load("outputs/uc_jepa/autoencoder/uc_ae_best.pt", map_location="cpu", weights_only=False)
mc = ck["config"]["model"]
model = UCAutoencoder(
    n_features=mc["n_features"], hidden_dim=mc["hidden_dim"], n_tokens=mc["n_tokens"],
    latent_dim=mc["latent_dim"], n_heads=mc["n_heads"], gru_layers=mc["gru_layers"],
    dropout=mc["dropout"], n_zones_max=mc["n_zones_max"], n_timesteps_max=mc["n_timesteps_max"],
    scenario_dim=mc.get("scenario_dim"),
    decoder_layers=int(mc.get("decoder_layers", 1)),
)
model.load_state_dict(ck["model_state_dict"]); model.eval()
USE_H = mc.get("scenario_dim") is not None

ds = UCLabelDataset(roots=["outputs/uc_jepa/dataset_train"], max_samples=400)
loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=uc_collate)

F = mc["n_features"]
thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
# (T, F) -> tp / fp / fn cumulative across batches
TP = np.zeros((len(thresholds), F)); FP = np.zeros((len(thresholds), F)); FN = np.zeros((len(thresholds), F))

with torch.no_grad():
    for b in loader:
        u = b["u_star"]; mask = b["zone_mask"]
        h = b.get("h_scenario") if USE_H else None
        probs = torch.sigmoid(model(u, zone_mask=mask, h_scenario=h)["logits"])
        m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(u)
        for ti, t in enumerate(thresholds):
            pb = (probs >= t).float()
            for f in range(F):
                pf = pb[..., f]; uf = u[..., f]; mf = m[..., f]
                TP[ti, f] += float(((pf == 1) & (uf == 1) & (mf == 1)).sum())
                FP[ti, f] += float(((pf == 1) & (uf == 0) & (mf == 1)).sum())
                FN[ti, f] += float(((pf == 0) & (uf == 1) & (mf == 1)).sum())

print(f"{'feature':<22} " + "  ".join(f"th={t:.1f}" for t in thresholds) + "   best")
for f in range(F):
    f1s = []
    for ti in range(len(thresholds)):
        pr = TP[ti, f] / max(1, TP[ti, f] + FP[ti, f])
        rc = TP[ti, f] / max(1, TP[ti, f] + FN[ti, f])
        f1s.append(2 * pr * rc / max(1e-9, pr + rc))
    best_t = thresholds[int(np.argmax(f1s))]
    print(f"{UC_FEATURE_NAMES[f]:<22} " + "  ".join(f"{x:>5.3f}" for x in f1s)
          + f"   {max(f1s):.3f}@{best_t}")

print("\nMacro F1 with shared threshold:")
for ti, t in enumerate(thresholds):
    f1s = []
    for f in range(F):
        pr = TP[ti, f] / max(1, TP[ti, f] + FP[ti, f])
        rc = TP[ti, f] / max(1, TP[ti, f] + FN[ti, f])
        f1s.append(2 * pr * rc / max(1e-9, pr + rc))
    print(f"  th={t:.1f}  macroF1={np.mean(f1s):.3f}")
