"""UC-JEPA-MILP hybrid pipeline.

Implemented pieces:
    - data.extract_uc_labels: materialize (u_star, h_scenario, family, criticality) per scenario.
    - models.uc_autoencoder: UC autoencoder with strategy-token latent.
    - models.scenario_to_uc_latent: h_scenario -> z_uc predictor (JEPA-style).
    - models.latent_ebm: energy model on z_uc conditioned by JEPA + scenario context.
    - sampling.latent_langevin: latent-space Langevin dynamics with best-of-K sampling.
    - workers.milp_warm_worker: thin wrapper around LPWorkerTwoStage for decoded candidates.
    - training.train_autoencoder: train the UC autoencoder.
    - training.train_jepa: train the scenario -> latent predictor.
    - training.train_latent_ebm: train the latent EBM on frozen AE + JEPA latents.
    - evaluation.sanity_check_ae: AE reconstruction + optional LP worker check.
    - evaluation.eval_jepa: direct JEPA decode evaluation.
    - evaluation.eval_latent_ebm: JEPA direct vs latent EBM + latent SGLD evaluation.
"""

# Number of UC binary features (F).
# Existing pipeline (src/ebm/dataset_v3.py) uses F=7. We extend to F=8 by adding
# `import_mode` derived from net_import[t] > 0. Index ordering below MUST stay
# stable: any new model artifact relies on it.
FEAT_BATT_CHARGE = 0
FEAT_BATT_DISCHARGE = 1
FEAT_PUMP_CHARGE = 2
FEAT_PUMP_DISCHARGE = 3
FEAT_DR = 4
FEAT_THERMAL_SU = 5
FEAT_THERMAL = 6
FEAT_IMPORT = 7
N_UC_FEATURES = 8

UC_FEATURE_NAMES = [
    "battery_charge_mode",
    "battery_discharge_mode",
    "pumped_charge_mode",
    "pumped_discharge_mode",
    "dr_active",
    "thermal_startup",
    "thermal_on",
    "import_mode",
]
