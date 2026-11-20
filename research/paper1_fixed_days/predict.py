"""Apply the paper's frozen pre-solve models to a fixed-grid operating day."""
from hashlib import sha256
import json
from pathlib import Path

from research.paper1_revision.realized.models import predict
from research.paper1_revision.realized.storage import DEST


TARGETS = ("equal4", "cost_priority", "co2_priority", "time_priority", "service4_co2", "E", "ES")
TRIO_COLUMNS = ("real_B1", "real_B3", "real_B4")
DIMENSION_COLUMNS = (
    "real_B1", "real_B2", "audit_lines", "audit_variables",
    "audit_free_binary_variables", "audit_active_constraints",
)


def load_frozen(archive: Path = DEST) -> tuple[dict, str]:
    """Read and authenticate the single coefficient archive used by this pilot."""
    freeze = json.loads((archive / "freeze.json").read_text(encoding="utf-8-sig"))
    raw = (archive / "frozen_models.json").read_bytes()
    fingerprint = sha256(raw).hexdigest()
    if fingerprint != freeze["models_sha256"]:
        raise ValueError("Frozen model coefficients differ from the recorded archive.")
    return json.loads(raw), fingerprint


def predict_day(models: dict, day_features: dict, budgets) -> list[dict]:
    """Return raw and bounded predictions for each target, representation and deadline."""
    structural = dict(zip(TRIO_COLUMNS, day_features["compact_trio"], strict=True))
    dimensions = dict(zip(DIMENSION_COLUMNS, day_features["model_dimensions"], strict=True))
    realized = day_features["realized_features"]
    if (any(structural[c] != realized[c] for c in TRIO_COLUMNS)
            or dimensions["real_B1"] != realized["real_B1"]
            or dimensions["real_B2"] != realized["real_B2"]
            or dimensions["audit_free_binary_variables"] != realized["real_B3"]):
        raise ValueError("Compact feature counts disagree with the realized-descriptor audit.")

    representations = {"realized_compact": structural, "model_size": dimensions, "realized": realized}
    rows = []
    for budget in budgets:
        for target in TARGETS:
            kind = "logistic" if target in {"E", "ES"} else "positive"
            for representation, features in representations.items():
                specification = models[f"{target}/{representation}/{kind}"]
                if str(budget) not in specification:
                    raise ValueError(f"No frozen predictor for the {budget}-second deadline.")
                model = specification[str(budget)]
                values = [features[column] for column in model["columns"]]
                raw = float(predict(model, [values])[0])
                rows.append({
                    "budget": budget, "target": target, "representation": representation,
                    "prediction_raw": raw, "prediction": min(1.0, max(0.0, raw)),
                })
    return rows
