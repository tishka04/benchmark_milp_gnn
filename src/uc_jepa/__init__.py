"""UC-JEPA-MILP hybrid pipeline.

Roadmap:
    Scenario graph -> HTE -> latent UC strategy -> UC decoder -> MILP worker (warm-started).

Phase 1 MVP (this package, current state):
    - data.extract_uc_labels:    materialize (u_star, h_scenario, family, criticality) per scenario.
    - models.uc_autoencoder:     UC autoencoder with strategy-token latent (variant B).
    - training.train_autoencoder: train the UC autoencoder.
    - evaluation.sanity_check_ae: BCE / Hamming / startup-consistency + optional LP worker check.

Later phases (not yet implemented):
    - models.scenario_to_uc_latent : h_scenario -> z_uc predictor.
    - models.latent_ebm            : energy on z_uc | h_scenario.
    - sampling.latent_langevin     : Langevin in latent space + best-of-K.
    - workers.milp_warm_worker     : unified hard / soft / local-repair wrapper around LPWorkerTwoStage.
    - evaluation.evaluate_uc_jepa  : full ablation suite.
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
