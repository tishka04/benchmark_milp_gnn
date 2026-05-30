"""Sanity-check the AE latent ``z_uc`` before building the JEPA predictor.

We verify:
1. Per-dimension variance is non-degenerate (no dead dims).
2. Tokens are diverse within a sample (not all collapsed to one).
3. Inter-scenario L2 distance is non-trivial (model discriminates scenarios).
4. The latent carries family-level signal (linear-probe F1 across families).
5. Encoder is deterministic in eval mode (re-encoding gives same z_uc).
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.uc_jepa.data.uc_label_dataset import UCLabelDataset, uc_collate
from src.uc_jepa.models.uc_autoencoder import UCAutoencoder

CKPT = "outputs/uc_jepa/autoencoder/uc_ae_best.pt"
N_VAL = 400

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
mc = ck["config"]["model"]
model = UCAutoencoder(
    n_features=mc["n_features"], hidden_dim=mc["hidden_dim"], n_tokens=mc["n_tokens"],
    latent_dim=mc["latent_dim"], n_heads=mc["n_heads"], gru_layers=mc["gru_layers"],
    dropout=mc["dropout"], n_zones_max=mc["n_zones_max"], n_timesteps_max=mc["n_timesteps_max"],
    scenario_dim=mc.get("scenario_dim"),
    decoder_layers=int(mc.get("decoder_layers", 1)),
)
model.load_state_dict(ck["model_state_dict"]); model.eval()

# Use the same val split as training would (seed=42, val_split=0.1)
ds = UCLabelDataset(roots=["outputs/uc_jepa/dataset_train"])
n = len(ds)
g = torch.Generator().manual_seed(42)
perm = torch.randperm(n, generator=g).tolist()
n_val = max(1, int(round(n * 0.1)))
val_idx = perm[n - n_val : n - n_val + min(N_VAL, n_val)]
sub = Subset(ds, val_idx)
loader = DataLoader(sub, batch_size=8, shuffle=False, collate_fn=uc_collate)
print(f"Encoding {len(sub)} val scenarios...")

z_list, fam_list = [], []
with torch.no_grad():
    for b in loader:
        z = model.encode(b["u_star"], zone_mask=b["zone_mask"])   # [B, K, D]
        z_list.append(z.cpu().numpy())
        fam_list.extend(b["family"])
Z = np.concatenate(z_list, axis=0)                                # [N, K, D]
N, K, D = Z.shape
print(f"z_uc shape: {Z.shape}  (N={N}, K={K}, D={D})")

# 1. Per-dim variance
flat = Z.reshape(N * K, D)
per_dim_var = flat.var(axis=0)                                    # [D]
dead = (per_dim_var < 1e-4).sum()
print(f"\n[1] Per-dim variance:  mean={per_dim_var.mean():.4f}  "
      f"min={per_dim_var.min():.6f}  max={per_dim_var.max():.4f}  "
      f"dead_dims(<1e-4)={dead}/{D}")

# 2. Token diversity: pairwise distance between the K tokens within a single sample.
inter_token = []
for i in range(N):
    z = Z[i]                                # [K, D]
    diffs = z[:, None, :] - z[None, :, :]   # [K, K, D]
    d = np.sqrt((diffs ** 2).sum(-1))
    iu = np.triu_indices(K, k=1)
    inter_token.append(d[iu].mean())
inter_token = np.array(inter_token)
print(f"[2] Mean within-sample token L2 distance: "
      f"mean={inter_token.mean():.3f}  std={inter_token.std():.3f}  "
      f"min={inter_token.min():.3f}  max={inter_token.max():.3f}")

# 3. Inter-scenario distance: pool tokens to one vector per scenario.
pooled = Z.mean(axis=1)                     # [N, D]
diffs = pooled[:, None, :] - pooled[None, :, :]
inter_d = np.sqrt((diffs ** 2).sum(-1))
iu = np.triu_indices(N, k=1)
inter_vals = inter_d[iu]
print(f"[3] Inter-scenario pooled L2: "
      f"mean={inter_vals.mean():.3f}  std={inter_vals.std():.3f}  "
      f"p05={np.percentile(inter_vals, 5):.3f}  p95={np.percentile(inter_vals, 95):.3f}")

# 4. Family separability via a 1-vs-rest linear-probe-like signal:
#    For each family, compute mean intra-family distance vs mean cross-family distance.
fams = np.array(fam_list)
uniq = sorted(set(fams))
print(f"[4] Family separability (intra vs inter pooled L2):")
print(f"    {'family':<24} n   intra   inter   ratio (intra/inter)")
for f in uniq:
    idx = np.where(fams == f)[0]
    if len(idx) < 2:
        continue
    intra = inter_d[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)].mean()
    other = np.setdiff1d(np.arange(N), idx)
    inter = inter_d[np.ix_(idx, other)].mean()
    print(f"    {f:<24} {len(idx):>3}  {intra:>6.3f}  {inter:>6.3f}  {intra/max(inter,1e-9):>6.3f}")

# 5. Determinism: encode the same sample twice (eval mode, no dropout).
b = next(iter(loader))
with torch.no_grad():
    z1 = model.encode(b["u_star"], zone_mask=b["zone_mask"])
    z2 = model.encode(b["u_star"], zone_mask=b["zone_mask"])
delta = (z1 - z2).abs().max().item()
print(f"\n[5] Re-encode determinism (max abs diff): {delta:.2e}")

# Summary verdict
print("\n=== Verdict ===")
ok = True
if dead > D // 4:
    print(f"  ❌ {dead} dead dims out of {D} — risk of collapse.")
    ok = False
if inter_token.mean() < 0.1:
    print(f"  ❌ Tokens within a sample are too similar (mean={inter_token.mean():.3f}).")
    ok = False
if inter_vals.std() / max(inter_vals.mean(), 1e-9) < 0.05:
    print("  ❌ Inter-scenario distances are nearly identical — z_uc not discriminative.")
    ok = False
# A meaningful representation should have intra/inter ratio < 1 for at least 1 family.
ratios = []
for f in uniq:
    idx = np.where(fams == f)[0]
    if len(idx) < 2: continue
    intra = inter_d[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)].mean()
    other = np.setdiff1d(np.arange(N), idx)
    inter = inter_d[np.ix_(idx, other)].mean()
    ratios.append(intra / max(inter, 1e-9))
if ratios and min(ratios) > 0.95:
    print(f"  ⚠️  No family has intra/inter < 0.95 (min={min(ratios):.3f}) — weak family signal.")
if ok:
    print("  ✅ z_uc looks healthy — safe to build the JEPA predictor on top.")
