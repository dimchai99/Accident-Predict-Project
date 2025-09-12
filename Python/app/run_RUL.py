# app/benchmark_health.py
'''
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
import os
import argparse

FEATURES = None
TRAIN_MEDIANS = None
THRESHOLD_INFO = 0.002300092949787481
THR = 0.002300092949787481

# app.db 연결 재사용
ROOT = Path(__file__).resolve().parents[1]  # .../Accident-Predict-Project/python
sys.path.insert(0, str(ROOT))
from app.db import get_cursor
import os
import argparse

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


results_s1, df_s1 = simple_preprocess_and_run(df_sample1)

results_s2, df_s2 = simple_preprocess_and_run(df_sample2)

results_s3, df_s3 = simple_preprocess_and_run(df_sample3)

results_s4, df_s4 = simple_preprocess_and_run(df_sample4)

results_s5, df_s5 = simple_preprocess_and_run(df_sample5)

results_s6, df_s6 = simple_preprocess_and_run(df_sample6)

SAMPLE_MAP = {
    (1, 0):   (results_s1, df_sample1),
    (1, 224): (results_s2, df_sample2),
    (2, 0):   (results_s3, df_sample3),
    (2, 120): (results_s4, df_sample4),
    (3, 0):   (results_s5, df_sample5),
    (3, 69):  (results_s6, df_sample6),
}

# ------------------------------
# 1) ECDF (mid-rank) 생성
# ------------------------------
def make_ecdf_midrank(baseline_vals: np.ndarray):
    arr = np.sort(np.asarray(baseline_vals, dtype=float))
    n = len(arr)
    if n == 0:
        raise ValueError("Baseline is empty.")
    def ecdf(x: float) -> float:
        r = np.searchsorted(arr, x, side="right")  # 0..n
        return float(r - 0.5) / float(n)           # mid-rank로 0/1 극단 완화
    return ecdf

# ------------------------------
# 2) 일자 baseline → ECDF 만들기
#    (daily_mse_df는 컬럼 ['date', 'recon_error_mse'] 가정)
# ------------------------------
def build_ecdf_from_daily(daily_mse_df: pd.DataFrame, baseline_days: int = 20):
    df = daily_mse_df.copy()
    # 날짜 안전 파싱
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "recon_error_mse"])
    df = df.sort_values("date")

    # baseline 구간 선택 + winsorize로 극단치 완화
    base = df["recon_error_mse"].iloc[:min(baseline_days, len(df))].to_numpy()
    base = pd.Series(base).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(base) == 0:
        raise ValueError("No valid values in baseline to build ECDF.")
    q_lo, q_hi = np.quantile(base, [0.01, 0.99])
    base = np.clip(base, q_lo, q_hi)

    return make_ecdf_midrank(base)

# ------------------------------
# 3) 최근 추세 기울기(a) 추정 (일 단위)
# ------------------------------
def fit_daily_health_slope(daily_mse_df: pd.DataFrame, window_days: int = 30, ema_alpha: float = 0.2):
    """
    - daily_mse_df: ['date','recon_error_mse','health'] 중 최소 ['date','health'] 필요
    - health는 0~100 범위 가정. 필요시 미리 계산되어 있어야 함.
    - window_days 내에서 1차 회귀로 일당 감소 기울기 a 추정
    """
    df = daily_mse_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "health"]).sort_values("date")

    if len(df) < 2:
        return np.nan

    # (선택) EMA로 health 스무딩 후 회귀 안정화
    df["health_smooth"] = df["health"].ewm(alpha=ema_alpha, adjust=False).mean()

    # 최근 window만 사용
    t = df.tail(min(window_days, len(df))).copy()
    if len(t) < 2:
        return np.nan

    x = (t["date"] - t["date"].min()).dt.days.to_numpy(dtype=float)
    y = t["health_smooth"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan

    a, b = np.polyfit(x, y, 1)  # y = a x + b
    return float(a)  # health/day

# ------------------------------
# 4) run 데이터 1회 → health_now 계산
#    (run_df는 2048행, 컬럼에 'recon_error_mse' 또는 샘플별 mse가 있다고 가정)
# ------------------------------
def health_from_run(run_df: pd.DataFrame, ecdf, mse_col: str = "recon_error_mse", agg: str = "mean"):
    """
    run 하나(8초, 2048행)의 recon_error_mse 요약치 → ECDF → health_now
    agg: 'mean' | 'median' | 'p90' 등 선택 가능
    """
    vals = pd.to_numeric(run_df[mse_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        raise ValueError("No valid recon_error_mse in run_df.")
    if agg == "mean":
        run_mse = float(vals.mean())
    elif agg == "median":
        run_mse = float(vals.median())
    elif agg.lower() in ("p90", "p95"):
        q = 0.9 if agg.lower() == "p90" else 0.95
        run_mse = float(vals.quantile(q))
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    # health = 100*(1-ECDF(mse))
    h = 100.0 * (1.0 - ecdf(run_mse))
    h = np.sqrt(h/100) * 100
    h = np.clip(h, 5, 95)
    return {"run_mse": run_mse, "health_now": h}

# ------------------------------
# 5) health_now + 기울기 a → RUL 계산
# ------------------------------
def rul_from_health(health_now: float, slope_per_day: float, asof_date) -> dict:
    """
    slope_per_day(a) < 0일 때만 유한 RUL. 단위: days
    asof_date: 이 run이 측정된 기준 날짜 (문자열/타임스탬프 허용)
    """
    asof_date = pd.to_datetime(asof_date, errors="coerce")
    if pd.isna(asof_date):
        raise ValueError("Invalid asof_date")

    if not np.isfinite(slope_per_day) or slope_per_day >= 0:
        return {"rul_days": float("inf"), "pred_end_date": None}

    rul_days = float(health_now / (-slope_per_day))
    pred_end = asof_date + pd.Timedelta(days=rul_days)
    return {"rul_days": rul_days, "pred_end_date": pred_end}

# ------------------------------
# 6) 종합: 1회 run → 즉시 RUL 뽑기
# ------------------------------
def infer_rul_for_single_run(
        run_df: pd.DataFrame,
        run_date: str,                       # 예: "2025-01-04"  (월/일만 있다면 "2025-01-04 00:00:00"처럼 연도를 붙여주세요)
        daily_mse_df: pd.DataFrame,          # 과거 일자 데이터 (['date','recon_error_mse','health'] 혹은 최소 ['date','recon_error_mse'])
        baseline_days: int = 10,
        window_days: int = 30,
        mse_col: str = "recon_error_mse",
        agg: str = "mean",
        reuse_health_in_daily: bool = True
):
    """
    - daily_mse_df에 이미 health가 있으면 reuse_health_in_daily=True로 기울기 계산에 재사용
      없으면 ECDF를 만든 뒤 daily_mse_df에도 health를 붙여서 기울기를 계산
    - run_df에서 run_mse→health_now→RUL 계산
    """
    # 1) ECDF
    ecdf = build_ecdf_from_daily(daily_mse_df, baseline_days=baseline_days)

    # 2) run → health_now
    run_info = health_from_run(run_df, ecdf, mse_col=mse_col, agg=agg)
    health_now = run_info["health_now"]

    # 3) slope a
    if reuse_health_in_daily and "health" in daily_mse_df.columns:
        df_for_slope = daily_mse_df[["date", "health"]].copy()
    else:
        # daily에도 같은 ECDF로 health 붙이기
        df_for_slope = daily_mse_df.copy()
        df_for_slope["date"] = pd.to_datetime(df_for_slope["date"], errors="coerce")
        df_for_slope = df_for_slope.dropna(subset=["date", "recon_error_mse"])
        df_for_slope["health"] = 100.0 * (1.0 - df_for_slope["recon_error_mse"].apply(lambda v: ecdf(float(v))))
        df_for_slope["health"] = df_for_slope["health"].clip(0.0, 100.0)

    a = fit_daily_health_slope(df_for_slope, window_days=window_days, ema_alpha=0.2)

    # 4) RUL
    rul_info = rul_from_health(health_now, a, run_date)

    # 5) 결과 패키징
    return {
        "run_mse": run_info["run_mse"],
        "health_now": health_now,
        "slope_per_day": a,
        "rul_days": rul_info["rul_days"],
        "pred_end_date": rul_info["pred_end_date"],
    }

# app/run_RUL.py

def compute_rul_from_sample(mode: int, blade_id: int) -> dict:
    """
    SAMPLE_MAP에서 (mode, blade_id)에 해당하는 샘플을 사용해
    health_now, rul_days, pred_end_date를 계산하여 dict로 반환.
    """
    key = (int(mode), int(blade_id))
    if key not in SAMPLE_MAP:
        raise ValueError(f"No sample for mode={mode}, blade_id={blade_id}")

    # 1) 샘플 꺼내기
    results_df, df_raw = SAMPLE_MAP[key]
    print(results_df)

    # 2) 최근 구간(대략 2048 샘플)을 run_df로 사용
    run_df = results_df.tail(2048).copy()

    # 3) 일별 평균 MSE(daily_mse_df) 구성
    if mode == 1:
        daily = pd.DataFrame({
        "date": ["2025-11-07","2025-11-22","2025-11-28","2025-12-10","2025-12-25"],
        "recon_error_mse": [0.000421, 0.000370, 0.000433, 0.000531, 0.007167],
        "health": [45.0, 65.0, 45.0, 30.0, 10.0]})
    elif mode == 2:
        daily = pd.DataFrame({
        "date": ["2025-11-07","2025-11-22","2025-11-28","2025-12-10","2025-12-25"],
        "recon_error_mse": [0.000330, 0.001003, 0.000867, 0.000605, 0.001851],
        "health": [80.0, 14.0, 67.11, 50, 12]})
    elif mode == 3:
        daily = pd.DataFrame({
        "date": ["2025-11-07","2025-11-22","2025-11-28","2025-12-10","2025-12-25"],
        "recon_error_mse": [0.000421, 0.000370, 0.000433, 0.000531, 0.007167],
        "health": [35.0, 40.0, 35.0, 25.0, 5.0]})

    # 4) RUL 추정
    res = infer_rul_for_single_run(
        run_df=run_df,
        run_date=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        daily_mse_df=daily,
        baseline_days=10,
        window_days=30,
        mse_col="recon_error_mse",
        agg="mean",
    )

    # 5) 반환 포맷 정리 (날짜는 문자열로)
    pred = res.get("pred_end_date")
    pred_str = pd.to_datetime(pred).strftime("%Y-%m-%d") if pred is not None else None

    return {
        "health_now": float(res.get("health_now")) if res.get("health_now") is not None else None,
        "rul_days": (float(res.get("rul_days"))
                     if (res.get("rul_days") not in (None, float("inf")))
                     else None),
        "pred_end_date": pred_str,
    }

daily_mse_df = pd.DataFrame({
    "date": ["2025-11-07","2025-11-22","2025-11-28","2025-12-10","2025-12-25"],
    "recon_error_mse": [0.000421, 0.000370, 0.000433, 0.000531, 0.007167],
    "health": [35.0, 40.0, 35.0, 25.0, 5.0]
})
# (가정) run_df: 2048행, 8초 구간의 샘플별 'recon_error_mse' 열 보유
# run_date: 이 run이 찍힌 날 (예: "2025-01-04 12:00:00" 또는 "2025-01-04")

'''

'''
result = infer_rul_for_single_run(
    run_df=results_s1,
    run_date="2025-09-11",
    daily_mse_df=daily_mse_df,    # 위 예시 테이블
    baseline_days=10,
    window_days=30,
    mse_col="recon_error_mse",
    agg="mean"                    # 필요시 'median' 또는 'p90'
)
'''