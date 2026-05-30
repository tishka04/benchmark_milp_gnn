"""Quick diagnostic: is the UC autoencoder collapsing to predicting all zeros?"""
import sys, os
sys.path.insert(0, os.getcwd())
import torch, numpy as np
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
model.load_state_dict(ck["model_state_dict"])
model.eval()
USE_H = mc.get("scenario_dim") is not None

ds = UCLabelDataset(roots=["outputs/uc_jepa/dataset_train"], max_samples=200)
loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=uc_collate)

F = mc["n_features"]
true_sum = np.zeros(F); pred_sum = np.zeros(F)
tp = np.zeros(F); fp = np.zeros(F); fn = np.zeros(F)
tot = 0

with torch.no_grad():
    for b in loader:
        u = b["u_star"]; mask = b["zone_mask"]
        h = b.get("h_scenario") if USE_H else None
        logits = model(u, zone_mask=mask, h_scenario=h)["logits"]
        p = (torch.sigmoid(logits) >= 0.5).float()
        m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(u)
        tot += m[..., 0].sum().item()
        for f in range(F):
            true_sum[f] += (u[..., f] * m[..., f]).sum().item()
            pred_sum[f] += (p[..., f] * m[..., f]).sum().item()
            tp[f] += ((p[..., f] == 1) & (u[..., f] == 1) & (m[..., f] == 1)).sum().item()
            fp[f] += ((p[..., f] == 1) & (u[..., f] == 0) & (m[..., f] == 1)).sum().item()
            fn[f] += ((p[..., f] == 0) & (u[..., f] == 1) & (m[..., f] == 1)).sum().item()

header = f"{'Feature':<20} {'true_rate':>10} {'pred_rate':>10} {'precision':>10} {'recall':>10} {'F1':>8}"
print(header)
for f in range(F):
    pr = tp[f] / max(1.0, tp[f] + fp[f])
    rc = tp[f] / max(1.0, tp[f] + fn[f])
    f1 = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"{UC_FEATURE_NAMES[f]:<20} {true_sum[f]/tot:>10.4f} {pred_sum[f]/tot:>10.4f} {pr:>10.4f} {rc:>10.4f} {f1:>8.4f}")
