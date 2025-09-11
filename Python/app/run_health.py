# app/benchmark_health.py
import sys, traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
import pickle
import joblib
from typing import Optional, Tuple
import matplotlib.pyplot as plt

# app.db 연결 재사용
ROOT = Path(__file__).resolve().parents[1]  # .../Accident-Predict-Project/python
sys.path.insert(0, str(ROOT))
from app.db import get_cursor
import os
import argparse

FEATURES = None
TRAIN_MEDIANS = None
THRESHOLD_INFO = 0.002300092949787481
THR = 0.002300092949787481

def load_run_measurements_by_sensor(blade_id, mode) -> pd.DataFrame:
    if mode == 1:
        min_num = 0
        max_num = 4095
    elif mode == 2:
        min_num = 4096
        max_num = 8192
    elif mode ==3:
        min_num = 8193
        max_num = 12288
    else:
        min_num = -1
        max_num = -1
        print("error")
    q = """
        SELECT date, timestamp,
               cut_torque, cut_lag_error, cut_speed,
               film_speed, film_lag_error, blade_id, measurement_id
        FROM run_measurement
        WHERE blade_id = %s AND measurement_id BETWEEN %s AND %s
        ORDER BY timestamp ASC \
        """
    with get_cursor() as cur:
        cur.execute(q, (blade_id,min_num, max_num))
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    df["mode"] = mode
    return df

df_sample1 = load_run_measurements_by_sensor(0, 1)
df_sample2 = load_run_measurements_by_sensor(224,1)
df_sample3 = load_run_measurements_by_sensor(0,2)
df_sample4 = load_run_measurements_by_sensor(120,2)
df_sample5 = load_run_measurements_by_sensor(0,3)
df_sample6 = load_run_measurements_by_sensor(69,3)

scaler = joblib.load("models/Oneyear_scaler_model1.pkl")

scaler1 = joblib.load("models/Oneyear_scaler_model1.pkl")
model1  = load_model("models/Oneyear_autoencoder_model1.h5",compile=False)

scaler2 = joblib.load("models/Oneyear_scaler_model2.pkl")
model2  = load_model("models/Oneyear_autoencoder_model2.h5",compile=False)

scaler3 = joblib.load("models/Oneyear_scaler_model3.pkl")
model3  = load_model("models/Oneyear_autoencoder_model3.h5",compile=False)


def preprocess_for_ae(
        df_raw: pd.DataFrame,
        date_col: Optional[str] = None,
        mode=None) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.Series]]:
    if mode == 1:
        scaler = scaler1
    elif mode ==2:
        scaler = scaler2
    elif mode == 3:
        scaler = scaler3
    else:
        print("error")

    # 날짜 컬럼 분리(있으면)
    dates: Optional[pd.Series] = None
    df = df_raw.copy()
    if date_col and date_col in df.columns:
        dates = df[date_col].copy()
        df = df.drop(columns=[date_col])

    # FEATURES 순서 고정 (강력 권장)
    if FEATURES is not None:
        # 누락된 컬럼 생성
        missing = [c for c in FEATURES if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        # 불필요한 컬럼 제거
        df = df[FEATURES]

    # 숫자형 변환
    df = df.apply(pd.to_numeric, errors="coerce")

    # 결측치 대체
    if TRAIN_MEDIANS is not None:
        df = df.fillna(TRAIN_MEDIANS.reindex(df.columns))
    else:
        df = df.fillna(df.median(numeric_only=True))

    # 스케일링
    X_scaled = scaler.transform(df.values)

    return df, X_scaled, dates


# ----------------------------------------------------
# 3) reconstruction error 계산 함수 (3.9 호환)
#    metric: "mse" 또는 "mae"
# ----------------------------------------------------
def reconstruction_error(
        X_scaled: np.ndarray,
        metric: str = "mse",
        batch_size: int = 1024,
        mode=None
) -> np.ndarray:
    if mode == 1:
        model = model1
    elif mode ==2:
        model = model2
    elif mode == 3:
        model = model3
    X_hat = model.predict(X_scaled, verbose=0, batch_size=batch_size)
    diff = X_scaled - X_hat
    err = np.mean(np.square(diff), axis=1)
    return err

# ----------------------------------------------------
# 4) 전체 파이프라인: df_raw -> 결과 DataFrame (3.9 호환)
# ----------------------------------------------------
def run_recon(
        df_raw: pd.DataFrame,
        date_col: Optional[str] = None,
        metric: Optional[str] = None,
        mode=None
) -> pd.DataFrame:
    # metric 자동 결정: 임계값 파일이 있으면 그 기준에 맞추고, 없으면 "mse"
    if metric is None:
        metric = (THRESHOLD_INFO if THRESHOLD_INFO else "mse") or "mse"

    df_ready, X_scaled, dates = preprocess_for_ae(df_raw, date_col=date_col, mode=mode)
    err = reconstruction_error(X_scaled, metric=metric, mode=mode)

    out = pd.DataFrame({
        "recon_error_mse": err
    }, index=df_ready.index)

    # 임계값 기준 이상치 플래그
    if THRESHOLD_INFO is not None:
        thr = float(THRESHOLD_INFO)
        out["is_anomaly"] = (out.iloc[:, 0].values > thr).astype(int)

    if dates is not None and date_col is not None:
        out.insert(0, date_col, dates.values)


    return out

# ----------------------------------------------------
# 5) 사용 예시


def simple_preprocess_and_run(csv_path):

    df = csv_path
    mode = int(df["mode"].iloc[0])
    df_for_model = df.drop(["timestamp", "blade_id", "mode", "measurement_id"], axis=1)
    # run_recon 실행 (외부에 이미 정의되어 있어야 함)
    results = run_recon(df_for_model, date_col="date", metric=None, mode = mode)
    return results, df


def insert_run_risk_from_results(results_df: pd.DataFrame, original_df: pd.DataFrame):
    """
    results_df: ['recon_error_mse', 'is_anomaly', 'date', 'mode']
    original_df: ['measurement_id', 'blade_id', 'timestamp', 'date', ...]  ← timestamp 이미 있음
    """

    merged = pd.concat([original_df.reset_index(drop=True),
                        results_df.reset_index(drop=True)], axis=1)

    # 2) insert용 레코드 구성
    #    ✅ timestamp는 df_s1의 'timestamp' 컬럼 그대로 사용!
    rows = [
        (
            int(row["measurement_id"]),      # PK
            int(row["blade_id"]),
            float(row["timestamp"]),         # df_s1의 timestamp 그대로
            float(row["recon_error_mse"]),
            int(row["mode"]),
        )
        for _, row in merged.iterrows()
        if pd.notna(row["recon_error_mse"]) and pd.notna(row["timestamp"])
    ]

    if not rows:
        print("[run_risk] no rows to insert")
        return

    sql = """
          INSERT INTO run_risk (run_risk_id, blade_id, timestamp, mse, mode)
          VALUES (%s, %s, %s, %s, %s)
              ON DUPLICATE KEY UPDATE
                                   mse  = VALUES(mse),
                                   mode = VALUES(mode) \
          """

    # 3) DB insert
    with get_cursor() as cur:
        cur.executemany(sql, rows)
        cur.connection.commit()
        print(f"[run_risk] inserted/updated rows: {len(rows)}")

        for r in cur.fetchall():
            print("   ", r)
# =========================
# 예: 기존 파이프라인 끝부분에 배치
# =========================
# results_s1, df_s1 이 준비된 상태라고 가정
# results_s1: ['recon_error_mse', 'is_anomaly', 'date']
# df_s1    : ['blade_id', 'mode', 'date', ...]


results_s1, df_s1 = simple_preprocess_and_run(df_sample1)
#insert_run_risk_from_results(results_s1, df_s1)
#print(results_s1.head())

results_s2, df_s2 = simple_preprocess_and_run(df_sample2)
#insert_run_risk_from_results(results_s2, df_s2)

results_s3, df_s3 = simple_preprocess_and_run(df_sample3)
insert_run_risk_from_results(results_s4, df_s3)

results_s4, df_s4 = simple_preprocess_and_run(df_sample4)
#insert_run_risk_from_results(results_s3, df_s4)

results_s5, df_s5 = simple_preprocess_and_run(df_sample5)
#insert_run_risk_from_results(results_s5, df_s5)

results_s6, df_s6 = simple_preprocess_and_run(df_sample6)
#insert_run_risk_from_results(results_s6, df_s6)

'''
plt.figure(figsize=(12,6))

# Test 데이터
plt.plot(results_s5['recon_error_mse'], label='sample1 Reconstruction Error', alpha=0.7)
plt.plot(results_s6['recon_error_mse'], label='sample3 Reconstruction Error', alpha=0.7)

plt.axhline(y=THR, color='r', linestyle='--', label=f'Threshold ({THR:.4f})')

plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error")
plt.legend()
plt.grid(True)
plt.show()

'''
