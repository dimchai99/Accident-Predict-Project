#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_once.py

MySQL에서 최신 inference_request 1건을 읽어와서
- 스케일러/모델 로드 (Keras 3 로더)
- 스케일 적용 및 재구성 오차 계산
- Spring Boot 엔드포인트로 scaled / result 저장

요구 패키지:
  pip install numpy requests pymysql joblib tensorflow keras scikit-learn

권장 버전:
  tensorflow==2.17.1
  keras==3.3.3
  scikit-learn==1.6.1
  numpy==1.26.4

환경변수(선택):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS
  SPRING_BASE (기본 http://localhost:8080)
  SCALER_PATH, MODEL_PATH
  THRESHOLD_VALUE, THRESHOLD_NAME
  N_FEATURES (fallback 기대 길이; 기본 8)
"""

import os
import json
import sys
import traceback
from typing import Tuple, List, Any

import numpy as np
import requests
import pymysql
from joblib import load
# Keras 3 로더 사용 (tf.keras 말고 keras)
from keras.models import load_model


# -----------------------------
# 환경 변수
# -----------------------------
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "accident_db")
DB_USER = os.getenv("DB_USER", "root")  # 필요 시 변경
DB_PASS = os.getenv("DB_PASS", "1234")

SPRING_BASE = os.getenv("SPRING_BASE", "http://localhost:8080").rstrip("/")
API_SCALED = f"{SPRING_BASE}/api/infer/scaled"
API_RESULTS = f"{SPRING_BASE}/api/infer/results"

# 기본 경로는 Windows 프로젝트 구조 예시 (환경변수로 덮어쓰기 권장)
SCALER_PATH = os.getenv(
    "SCALER_PATH",
    r"C:\Users\1\Accident-Predict-Project\python\standard_scaler_v1.joblib"
)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"C:\Users\1\Accident-Predict-Project\python\autoencoder_v1.keras"
)

THRESHOLD_VALUE = float(os.getenv("THRESHOLD_VALUE", "0.5"))
THRESHOLD_NAME = os.getenv("THRESHOLD_NAME", "fixed_0.5")

# 기대 피처 길이 (모델/스케일러에서 자동 감지 실패 시만 사용)
N_FEATURES = int(os.getenv("N_FEATURES", "8"))


# -----------------------------
# DB 유틸
# -----------------------------
def get_mysql_conn():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def fetch_latest_request() -> dict:
    sql = """
          SELECT request_id, model_name, model_version,
                 scaler_name, scaler_version, features_json
          FROM inference_request
          ORDER BY created_at DESC
              LIMIT 1 \
          """
    conn = get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if not row:
                raise RuntimeError("inference_request 테이블에 데이터가 없습니다.")
            return row
    finally:
        conn.close()


# -----------------------------
# 모델/스케일러 로드
# -----------------------------
_scaler = None
_model = None

def load_assets():
    global _scaler, _model
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {SCALER_PATH}")
        _scaler = load(SCALER_PATH)
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        # Keras 3 로더
        _model = load_model(MODEL_PATH, compile=False)


# -----------------------------
# 전처리 & 추론
# -----------------------------
def ensure_list(obj: Any) -> List[float]:
    # pymysql DictCursor가 JSON 컬럼을 str로 주는 경우가 많음
    if isinstance(obj, str):
        return list(json.loads(obj))
    if isinstance(obj, (list, tuple, np.ndarray)):
        return list(obj)
    raise TypeError(f"features_json 타입을 처리할 수 없습니다: {type(obj)}")


def _detect_expected_dim() -> int:
    """모델 또는 스케일러에서 기대 입력 차원 자동 감지. 실패 시 N_FEATURES 반환."""
    expected = None
    try:
        if _model is not None and hasattr(_model, "input_shape"):
            # 일반적으로 (None, n_features) 형태
            shp = _model.input_shape
            if isinstance(shp, (list, tuple)) and len(shp) > 0:
                expected = shp[-1]
                if isinstance(expected, (list, tuple)):
                    expected = expected[-1]
    except Exception:
        expected = None

    if expected is None:
        try:
            if _scaler is not None and hasattr(_scaler, "n_features_in_"):
                expected = int(_scaler.n_features_in_)
        except Exception:
            expected = None

    if expected is None or not isinstance(expected, int):
        expected = N_FEATURES
    return expected


def run_inference(features: List[float]) -> Tuple[float, List[float], List[float]]:
    expected = _detect_expected_dim()

    # 길이 보정: 많으면 자르고, 부족하면 에러
    if len(features) != expected:
        if len(features) > expected:
            print(f"[warn] features length={len(features)} -> expected={expected}. trimming.")
            features = features[:expected]
        else:
            raise ValueError(f"features 길이가 {expected}가 아닙니다. 실제: {len(features)}")

    X = np.array([features], dtype=float)      # (1, n_features)
    X_scaled = _scaler.transform(X)            # (1, n_features)
    recon = _model.predict(X_scaled, verbose=0)

    sqerr = (X_scaled - recon) ** 2
    mse = float(np.mean(sqerr))
    per_feat_sqerr = sqerr.flatten().astype(float).tolist()
    scaled_values = X_scaled.flatten().astype(float).tolist()

    return mse, per_feat_sqerr, scaled_values


# -----------------------------
# Spring API 저장
# -----------------------------
def post_scaled(request_id: str, scaled_values: List[float]):
    payload = {
        "requestId": request_id,
        "scaledJson": json.dumps(scaled_values),
    }
    r = requests.post(API_SCALED, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def post_result(request_id: str, model_name: str, model_version: str,
                mse: float, per_feat_sqerr: List[float]):
    payload = {
        "requestId": request_id,
        "modelName": model_name,
        "modelVersion": model_version,
        "reconstructionMse": mse,
        "thresholdName": THRESHOLD_NAME,
        "thresholdValue": THRESHOLD_VALUE,
        "isAnomaly": bool(mse >= THRESHOLD_VALUE),
        "detailJson": json.dumps({"per_feature_sqerr": per_feat_sqerr}),
    }
    r = requests.post(API_RESULTS, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# -----------------------------
# main
# -----------------------------
def main():
    print("[*] 최신 요청 1건 조회")
    row = fetch_latest_request()
    request_id = row["request_id"]
    model_name = row["model_name"]
    model_version = row["model_version"]
    features = ensure_list(row["features_json"])

    print(f"    request_id={request_id}")
    print(f"    model={model_name}/{model_version}")
    print(f"    features(len={len(features)}): {features}")

    print("[*] 스케일러/모델 로드")
    load_assets()

    print("[*] 추론 실행")
    mse, per_feat_sqerr, scaled_values = run_inference(features)
    print(f"    MSE={mse:.6f}  (threshold={THRESHOLD_VALUE})  anomaly={mse >= THRESHOLD_VALUE}")

    print("[*] scaled_features 저장")
    try:
        scaled_resp = post_scaled(request_id, scaled_values)
        print("    scaled ->", scaled_resp)
    except Exception as e:
        # scaled 저장 실패는 치명적이지 않으니 경고만 출력하고 계속 진행
        print("    scaled 저장 실패(무시하고 진행):", repr(e))

    print("[*] inference_result 저장")
    res_resp = post_result(request_id, model_name, model_version, mse, per_feat_sqerr)
    print("    result ->", res_resp)
    print("[✓] 완료")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[!] 오류 발생")
        print(e)
        traceback.print_exc()
        sys.exit(1)
