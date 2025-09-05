#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_model_manifest_v2.py
- Adds --skip-model to proceed without loading the .keras file (helps when TF DLLs are missing).
- More robust loader: tries Keras 3's `keras.saving.load_model` first, then falls back to tf.keras.
"""

import argparse
import json
import os
import sys
import hashlib
import textwrap
from datetime import datetime

def sha256_of_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def coerce_name(name: str) -> str:
    import re
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    if not s:
        s = 'f'
    if s[0].isdigit():
        s = 'f_' + s
    reserved = {'select','from','where','group','order','by','limit','and','or','not'}
    if s in reserved:
        s = s + '_f'
    return s

def load_scaler_info(scaler_path):
    try:
        import joblib
    except Exception as e:
        print("ERROR: joblib/scikit-learn required to load the scaler:", e, file=sys.stderr)
        return None

    scaler = joblib.load(scaler_path)
    names = getattr(scaler, "feature_names_in_", None)
    n = getattr(scaler, "n_features_in_", None)
    info = {
        "class": scaler.__class__.__name__,
        "n_features_in_": int(n) if n is not None else None,
        "feature_names_in_": list(map(str, names)) if names is not None else None,
    }
    for attr in ["with_mean","with_std","quantile_range"]:
        if hasattr(scaler, attr):
            try:
                info[attr] = getattr(scaler, attr)
            except Exception:
                pass
    # Avoid dumping big arrays
    for attr in ["center_","scale_"]:
        if hasattr(scaler, attr):
            try:
                arr = getattr(scaler, attr)
                info[attr] = f"<array length {len(arr)}>"
            except Exception:
                pass
    return scaler, info

def load_model_info_any(model_path):
    """
    Try to load a .keras model using Keras 3 first; if not available, try tf.keras.
    Return (model_or_none, info_dict).
    """
    # Try Keras 3
    try:
        import keras
        try:
            model = keras.saving.load_model(model_path, compile=False)
            input_shapes = []
            try:
                if hasattr(model, "inputs"):
                    for t in model.inputs:
                        shape = getattr(t, "shape", None)
                        if shape is not None:
                            try:
                                input_shapes.append(tuple(int(d) if d is not None else None for d in shape))
                            except Exception:
                                input_shapes.append(tuple(shape))
            except Exception:
                pass

            # summary
            summary_lines = []
            try:
                model.summary(print_fn=lambda x: summary_lines.append(x))
            except Exception:
                summary_lines = ["(model.summary() not available)"]
            return model, {"backend": "keras3", "input_shapes": input_shapes, "summary": "\n".join(summary_lines[:2000])}
        except Exception as e:
            print("Keras 3 load failed, will try tf.keras next:", e, file=sys.stderr)
    except Exception as e:
        print("Keras import failed, will try tf.keras next:", e, file=sys.stderr)

    # Fallback to tf.keras
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shapes = []
        try:
            if hasattr(model, "inputs"):
                for t in model.inputs:
                    shape = getattr(t, "shape", None)
                    if shape is not None:
                        try:
                            input_shapes.append(tuple(int(d) if d is not None else None for d in shape))
                        except Exception:
                            input_shapes.append(tuple(shape))
        except Exception:
            pass
        summary_lines = []
        try:
            model.summary(print_fn=lambda x: summary_lines.append(x))
        except Exception:
            summary_lines = ["(model.summary() not available)"]
        return model, {"backend": "tf.keras", "input_shapes": input_shapes, "summary": "\n".join(summary_lines[:2000])}
    except Exception as e:
        print("Both Keras3 and tf.keras load attempts failed:", e, file=sys.stderr)
        return None, None

def write_manifest(outdir, payload):
    path = os.path.join(outdir, "model_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def write_mysql_schema(outdir, feature_names):
    if feature_names is None:
        ddl = textwrap.dedent(f"""
        -- JSON-based schema (fallback when feature names are unknown)
        CREATE TABLE IF NOT EXISTS inference_request (
          request_id      CHAR(36) PRIMARY KEY,
          model_name      VARCHAR(100) NOT NULL,
          model_version   VARCHAR(50)  NOT NULL,
          scaler_name     VARCHAR(100) NOT NULL,
          scaler_version  VARCHAR(50)  NOT NULL,
          source_system   VARCHAR(100) NULL,
          measurement_ts  DATETIME NULL,
          features_json   JSON NOT NULL,
          created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          INDEX idx_req_created_at (created_at)
        );

        CREATE TABLE IF NOT EXISTS inference_result (
          result_id       BIGINT PRIMARY KEY AUTO_INCREMENT,
          request_id      CHAR(36) NOT NULL,
          model_name      VARCHAR(100) NOT NULL,
          model_version   VARCHAR(50)  NOT NULL,
          reconstruction_mse  DOUBLE NOT NULL,
          threshold_name  VARCHAR(100) NULL,
          threshold_value DOUBLE NULL,
          is_anomaly      BOOLEAN NOT NULL,
          detail_json     JSON NULL,
          created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (request_id) REFERENCES inference_request(request_id) ON DELETE CASCADE,
          INDEX idx_res_request (request_id),
          INDEX idx_res_created_at (created_at)
        );
        """).strip()
    else:
        cols = [f"  `{coerce_name(n)}` DOUBLE NOT NULL" for n in feature_names]
        cols_sql = ",\n".join(cols)
        ddl = textwrap.dedent(f"""
        -- Column-based schema tailored to your feature names
        CREATE TABLE IF NOT EXISTS inference_request_wide (
          request_id      CHAR(36) PRIMARY KEY,
          model_name      VARCHAR(100) NOT NULL,
          model_version   VARCHAR(50)  NOT NULL,
          scaler_name     VARCHAR(100) NOT NULL,
          scaler_version  VARCHAR(50)  NOT NULL,
          source_system   VARCHAR(100) NULL,
          measurement_ts  DATETIME NULL,
{cols_sql},
          created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          INDEX idx_req_created_at (created_at)
        );

        CREATE TABLE IF NOT EXISTS inference_result (
          result_id       BIGINT PRIMARY KEY AUTO_INCREMENT,
          request_id      CHAR(36) NOT NULL,
          model_name      VARCHAR(100) NOT NULL,
          model_version   VARCHAR(50)  NOT NULL,
          reconstruction_mse  DOUBLE NOT NULL,
          threshold_name  VARCHAR(100) NULL,
          threshold_value DOUBLE NULL,
          is_anomaly      BOOLEAN NOT NULL,
          detail_json     JSON NULL,
          created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (request_id) REFERENCES inference_request_wide(request_id) ON DELETE CASCADE,
          INDEX idx_res_request (request_id),
          INDEX idx_res_created_at (created_at)
        );
        """).strip()
    path = os.path.join(outdir, "mysql_schema.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ddl + "\n")
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".keras model path")
    ap.add_argument("--scaler", required=True, help=".joblib scaler path")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--model-name", default="autoencoder_v1")
    ap.add_argument("--model-version", default="v1")
    ap.add_argument("--scaler-name", default="standard_scaler_v1")
    ap.add_argument("--scaler-version", default="v1")
    ap.add_argument("--skip-model", action="store_true", help="Skip loading the model (useful if TF DLLs are missing)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model_sha = sha256_of_file(args.model)
    scaler_sha = sha256_of_file(args.scaler)

    scaler_loaded, scaler_info = load_scaler_info(args.scaler)
    if scaler_loaded is None:
        sys.exit(2)

    feature_names = scaler_info.get("feature_names_in_")
    n_features = scaler_info.get("n_features_in_")

    model_info = None
    if not args.skip_model:
        model_loaded, model_info = load_model_info_any(args.model)
        if model_loaded is None:
            print("WARNING: Model could not be loaded. Continue without model details. You can re-run later.", file=sys.stderr)

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": {
            "name": args.model_name,
            "version": args.model_version,
            "framework": ("keras3/tf-unknown" if model_info is None else model_info.get("backend", "unknown")),
            "file_path": os.path.abspath(args.model),
            "sha256": model_sha,
            "input_shapes": None if model_info is None else model_info.get("input_shapes"),
            "summary_head": None if model_info is None else model_info.get("summary", "").splitlines()[:15]
        },
        "scaler": {
            "name": args.scaler_name,
            "version": args.scaler_version,
            "lib": "scikit-learn",
            "file_path": os.path.abspath(args.scaler),
            "sha256": scaler_sha,
            "class": scaler_info.get("class"),
            "n_features_in_": n_features,
            "feature_names_in_": feature_names
        }
    }

    manifest_path = write_manifest(args.outdir, manifest)
    schema_path = write_mysql_schema(args.outdir, feature_names)

    print("Wrote:")
    print("  -", manifest_path)
    print("  -", schema_path)
    print("\nDone. Share model_manifest.json with your teammate/ChatGPT to generate tailored SQL.")

if __name__ == "__main__":
    main()
