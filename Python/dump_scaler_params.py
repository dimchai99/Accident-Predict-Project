# dump_scaler_params.py
import joblib, json, numpy as np

IN = "standard_scaler_v1.joblib"
OUT = "standard_scaler_params.json"

scaler = joblib.load(IN)

params = {
    "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
    "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
    "var": getattr(scaler, "var_", None).tolist() if hasattr(scaler, "var_") else None,
    "n_features_in": int(getattr(scaler, "n_features_in_", len(scaler.scale_) if hasattr(scaler, "scale_") else 0)),
    "feature_names_in": scaler.feature_names_in_.tolist() if hasattr(scaler, "feature_names_in_") else None,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False)
print(f"Saved params to {OUT}")
