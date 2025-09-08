import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model


def load_scaler(scaler_path: str):
    """전처리 스케일러 로드"""
    if not scaler_path or not os.path.exists(scaler_path):
        print(f"[WARN] Scaler 파일 없음: {scaler_path}", file=sys.stderr)
        return None
    return joblib.load(scaler_path)


def run_inference(csv_path: str, model_path: str, scaler_path: str, feature_cols: list):
    # 1) CSV 불러오기
    df = pd.read_csv(csv_path)

    # 2) 필요한 컬럼 확인
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {col}")

    X = df[feature_cols].astype(float).values

    # 3) 스케일러 적용
    scaler = load_scaler(scaler_path)
    if scaler is not None:
        X = scaler.transform(X)

    # 4) 모델 로드
    model = load_model(model_path)

    # 5) 예측
    y_pred = model.predict(X, verbose=0)
    y_pred = np.ravel(y_pred)  # (n,) 형태로 변환

    # 6) 결과 JSON 생성
    result = {
        "num_samples": int(len(y_pred)),
        "predictions": y_pred.tolist(),
        "summary": {
            "mean": float(np.mean(y_pred)),
            "std": float(np.std(y_pred)),
            "min": float(np.min(y_pred)),
            "max": float(np.max(y_pred)),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="입력 CSV 파일 경로")
    parser.add_argument("--model", required=True, help="Keras 모델(.h5) 경로")
    parser.add_argument("--scaler", required=True, help="전처리 스케일러(.pkl) 경로")
    parser.add_argument(
        "--features",
        default="cut_torque,cut_lag_error,cut_speed,film_speed,film_lag_error",
        help="모델 입력 컬럼 이름들 (콤마 구분)",
    )
    parser.add_argument("--output", default=None, help="출력 JSON 파일 경로 (옵션)")
    args = parser.parse_args()

    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]

    try:
        result = run_inference(args.csv, args.model, args.scaler, feature_cols)
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        print(result_json)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result_json)

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
