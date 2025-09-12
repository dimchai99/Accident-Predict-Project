# app/benchmark_health.py
import sys, traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
import pickle
import joblib
from typing import Optional, Tuple
import os
import argparse

# app.db 연결 재사용
ROOT = Path(__file__).resolve().parents[1]  # .../Accident-Predict-Project/python
sys.path.insert(0, str(ROOT))
from app.db import get_cursor
import os
import argparse

# -----------------------------
# 0) 공통 설정 (MSE 고정 + 모드별 임계값)
# -----------------------------
THRESHOLDS = {
    1: 0.002300092949787481,
    2: 0.0007001939702707699,  # 모드별로 다르면 값 조정
    3: 0.0008032518997440094,
}
FEATURES = ["cut_torque", "cut_lag_error", "cut_speed", "film_speed", "film_lag_error"]
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
import os
import argparse

TRAIN_MEDIANS = None

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

def preprocess_for_ae(df_raw, date_col=None, mode=None):
    print(f"▶ preprocess 시작: df_raw.shape={df_raw.shape}, 컬럼={list(df_raw.columns)}")

    if mode == 1:
        scaler = scaler1
    elif mode == 2:
        scaler = scaler2
    elif mode == 3:
        scaler = scaler3

    df = df_raw.copy()

    if date_col and date_col in df.columns:
        df = df.drop(columns=[date_col])

    # FEATURES 순서 강제
    missing = [c for c in FEATURES if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    df = df[FEATURES]

    # 숫자형 변환
    df = df.apply(pd.to_numeric, errors="coerce")
    print("▶ 숫자 변환 후 NaN 개수:", df.isna().sum().to_dict())

    # 결측치 채우기
    df = df.fillna(df.median(numeric_only=True))
    print("▶ fillna 후 NaN 개수:", df.isna().sum().to_dict())

    # 스케일링
    X_scaled = scaler.transform(df.values)
    print("▶ X_scaled 예시:", X_scaled[:5])

    return df, X_scaled, None

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
    # metric 자동 결정
    if metric is None:
        metric = "mse"

    df_ready, X_scaled, dates = preprocess_for_ae(df_raw, date_col=date_col, mode=mode)
    err = reconstruction_error(X_scaled, metric=metric, mode=mode)

    out = pd.DataFrame({"recon_error_mse": err}, index=df_ready.index)

    # --- 모드별 임계값 적용 ---
    if mode is not None and int(mode) in THRESHOLDS:
        thr = THRESHOLDS[int(mode)]
        print(f"✅ run_recon 적용된 threshold({mode}) = {thr}")
        out["is_anomaly"] = (out["recon_error_mse"].values > thr).astype(int)
    else:
        print(f"⚠️ run_recon: mode={mode} 에 대한 threshold 없음, anomaly=0 처리")
        out["is_anomaly"] = 0

    if dates is not None and date_col is not None:
        out.insert(0, date_col, dates.values)

    return out

# ----------------------------------------------------
# 5) 사용 예시


def simple_preprocess_and_run(csv_path):

    df = csv_path
    print(df.head())
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

print(results_s1)

def make_daily_mse(
        results: pd.DataFrame,
        default_year: int = 2025,
        after: str = None,                  # 예: "2025-06-16"
        between: tuple = None,              # 예: ("2025-06-16", "2025-12-31")
        exclude_dates: list = None          # 예: ["2025-06-15","2025-06-16"]
) -> pd.DataFrame:
    if "date" not in results.columns or "recon_error_mse" not in results.columns:
        raise ValueError("results에는 'date'와 'recon_error_mse' 컬럼이 있어야 합니다.")

    df = results.copy()
    df["recon_error_mse"] = pd.to_numeric(df["recon_error_mse"], errors="coerce")
    df = df.dropna(subset=["recon_error_mse"])

    s = df["date"].astype(str).str.strip()
    has_year = s.str.match(r"\d{4}-\d{2}-\d{2}")
    s_full = s.copy()
    s_full.loc[~has_year] = f"{default_year}-" + s.loc[~has_year]

    # 시:분:초 있는 케이스 우선
    dt = pd.to_datetime(s_full, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    mask_nat = dt.isna()
    if mask_nat.any():
        # 날짜만 있는 케이스 보완
        dt2 = pd.to_datetime(s_full[mask_nat], format="%Y-%m-%d", errors="coerce")
        dt.loc[mask_nat] = dt2

    if dt.isna().all():
        print("[DEBUG] date 파싱 실패 예시:", s.head().tolist())
        return pd.DataFrame(columns=["date", "recon_error_mse"])

    df["date"] = dt.dt.date
    df = df.dropna(subset=["date"])

    # ---- 여기서 '자르기'를 진짜로 적용 ----
    if after:
        df = df[pd.to_datetime(df["date"]) > pd.to_datetime(after)]
    if between:
        start, end = between
        df = df[(pd.to_datetime(df["date"]) >= pd.to_datetime(start)) &
                (pd.to_datetime(df["date"]) <= pd.to_datetime(end))]
    if exclude_dates:
        ex = set(pd.to_datetime(pd.Series(exclude_dates)).dt.date.tolist())
        df = df[~df["date"].isin(ex)]
    # -------------------------------------

    daily_mse = (
        df.groupby("date", as_index=False)["recon_error_mse"]
        .mean()
        .rename(columns={"recon_error_mse": "recon_error_mse"})
    )

    return daily_mse
def add_health_minmax(
        daily_mse_df: pd.DataFrame,
        global_min: float = None,
        global_max: float = None,
        clip_quantiles: tuple = (0.01, 0.99),   # <-- 추가
        log1p: bool = True
) -> tuple[pd.DataFrame, dict]:
    df = daily_mse_df.copy()
    df["recon_error_mse"] = pd.to_numeric(df["recon_error_mse"], errors="coerce")
    df = df.dropna(subset=["recon_error_mse"])

    vals = df["recon_error_mse"].to_numpy()

    # 1) 윈저라이즈: 극단값을 분위수로 잘라서 스케일 왜곡 방지
    if clip_quantiles is not None:
        qlo, qhi = np.quantile(vals, clip_quantiles)
        vals_clip = np.clip(vals, qlo, qhi)
    else:
        vals_clip = vals

    # 2) 스케일 구간 결정 (글로벌 제공 없으면 클리핑된 분포에서 결정)
    if global_min is None or global_max is None:
        mse_min = float(np.min(vals_clip))
        mse_max = float(np.max(vals_clip))
    else:
        mse_min, mse_max = float(global_min), float(global_max)

    # 3) log1p + min–max
    eps = 1e-18
    if log1p:
        lo, hi = np.log1p(max(mse_min,0.0)), np.log1p(max(mse_max,0.0))
        span = max(hi - lo, eps)
        def mapper(v):
            z = (hi - np.log1p(max(float(v),0.0))) / span
            return float(np.clip(100.0*z, 0.0, 100.0))
    else:
        lo, hi = max(mse_min,0.0), max(mse_max,0.0)
        span = max(hi - lo, eps)
        def mapper(v):
            z = (hi - max(float(v),0.0)) / span
            return float(np.clip(100.0*z, 0.0, 100.0))

    df["health"] = df["recon_error_mse"].apply(mapper)

    return df, {"mse_min": mse_min, "mse_max": mse_max, "clip_q": clip_quantiles, "log1p": log1p}
# --- 1) 분위수 + 감마 보정 헬스 매퍼 ---
def build_health_quantile_mapper(
        baseline_vals: np.ndarray,
        p_low: float = 0.05,
        p_high: float = 0.95,
        gamma: float = 0.7,
        use_log: bool = True
):
    """
    baseline 분포의 [p_low, p_high] 구간을 100→0으로 선형 매핑하고,
    z ** gamma 로 비선형 보정( gamma<1 이면 상단 포화 완화 ).
    """
    base = np.asarray(baseline_vals, dtype=float)
    base = pd.Series(base).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if base.size == 0:
        raise ValueError("Empty baseline for health mapping.")

    base_t = np.log1p(base) if use_log else base
    lo = np.quantile(base_t, p_low)
    hi = np.quantile(base_t, p_high)
    eps = 1e-12
    span = max(hi - lo, eps)

    def mapper(mse_val: float) -> float:
        v = float(mse_val)
        v = np.log1p(v) if use_log else v
        # hi(나쁨)에서 0, lo(좋음)에서 1
        z = (hi - v) / span
        z = np.clip(z, 0.0, 1.0)
        if gamma != 1.0:
            z = z ** gamma
        return float(100.0 * z)

    return mapper, {"p_low": p_low, "p_high": p_high, "gamma": gamma, "use_log": use_log, "lo": float(lo), "hi": float(hi)}

# --- 2) DF에 health 붙이기 (공통 스케일 옵션) ---
def add_health_quantile(
        daily_mse_df: pd.DataFrame,
        p_low: float = 0.05,
        p_high: float = 0.95,
        gamma: float = 0.7,
        use_log: bool = True,
        # 3.9 호환: Union 사용
        global_minmax_from: Optional[Union[pd.Series, np.ndarray]] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    daily_mse_df: ['date','recon_error_mse'] 필요.
    - 기본: 현재 DF의 분포를 baseline으로 사용
    - 여러 모드를 같은 스케일로 보고 싶으면 global_minmax_from(여러 DF의 recon_error_mse concat)을 넘겨
      그 분포로 분위수를 계산
    """
    df = daily_mse_df.copy()
    df["recon_error_mse"] = pd.to_numeric(df["recon_error_mse"], errors="coerce")
    df = df.dropna(subset=["recon_error_mse"])

    # baseline 선택
    if global_minmax_from is not None:
        base_vals = pd.to_numeric(pd.Series(global_minmax_from), errors="coerce").dropna().to_numpy()
    else:
        base_vals = df["recon_error_mse"].to_numpy()

    mapper, calib = build_health_quantile_mapper(
        base_vals, p_low=p_low, p_high=p_high, gamma=gamma, use_log=use_log
    )
    df["health"] = df["recon_error_mse"].apply(lambda v: mapper(float(v)))

    return df, calib

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict

# =========================
# 0) 스케일 정보 → health mapper
# =========================
def build_health_mapper_from_scaleinfo(scale_info: Dict):
    """
    scale_info 예:
      {'p_low':0.05,'p_high':0.95,'gamma':0.7,'use_log':True,'lo':..., 'hi':...}
    반환: mse_val -> health(0~100)
    """
    p_low  = scale_info.get("p_low", 0.05)
    p_high = scale_info.get("p_high", 0.95)
    gamma  = scale_info.get("gamma", 1.0)
    use_log = bool(scale_info.get("use_log", True))
    lo = float(scale_info["lo"])
    hi = float(scale_info["hi"])
    eps = 1e-18
    span = max(hi - lo, eps)

    def mapper(mse_val: float) -> float:
        v = float(mse_val)
        v = np.log1p(v) if use_log else v
        # lo(좋음)→1, hi(나쁨)→0
        z = (hi - v) / span
        z = np.clip(z, 0.0, 1.0)
        # 감마 보정( <1 이면 상단 눌림 )
        if gamma != 1.0:
            z = z ** gamma
        return float(100.0 * z)

    return mapper

# =========================
# 1) 일별 health 기울기(slope) 추정
# =========================
def fit_daily_health_slope(daily_df: pd.DataFrame, window_days: int = 30, ema_alpha: float = 0.2) -> float:
    """
    daily_df: ['date','health'] 포함. date는 날짜/문자열 모두 허용
    반환: a (점/일)
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "health"]).sort_values("date")
    if len(df) < 2:
        return float("nan")

    df["health_smooth"] = df["health"].ewm(alpha=ema_alpha, adjust=False).mean()
    t = df.tail(min(window_days, len(df)))
    if len(t) < 2:
        return float("nan")

    x = (t["date"] - t["date"].min()).dt.days.to_numpy(dtype=float)
    y = t["health_smooth"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return float("nan")

    a, b = np.polyfit(x[m], y[m], 1)  # y = a x + b
    return float(a)

# =========================
# 2) RUL 계산 (보수화 옵션 포함)
# =========================
def compute_rul_days_refined(
        daily_df: pd.DataFrame,
        window_days: int = 30,
        health_now: Optional[float] = None,          # 샘플의 health_now (없으면 최신 health 사용)
        asof_date: Optional[Union[str, pd.Timestamp]] = None,
        health_col: str = "health",
        health_end: float = 20.0,          # 교체 임계 건강도(조정 가능)
        slope_floor: float = -0.2,         # 최소 감소율(점/일, 더 음수로 보수화)
        max_rul_days: Optional[float] = 365.0        # RUL 상한
) -> Dict[str, object]:
    """
    daily_df: ['date', health_col] 필요
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", health_col]).sort_values("date")
    if len(df) == 0:
        return {"slope": float("nan"), "rul_days": float("inf"), "pred_end_date": None}

    # slope
    a = fit_daily_health_slope(
        df[["date", health_col]].rename(columns={health_col: "health"}),
        window_days=window_days,
        ema_alpha=0.2,
    )

    # 기준 날짜/health_now
    last_date = df["date"].iloc[-1] if asof_date is None else pd.to_datetime(asof_date, errors="coerce")
    h_now = float(df[health_col].iloc[-1]) if health_now is None else float(health_now)

    # 증가(개선) 중이거나 유효치 없으면 무한대
    if not np.isfinite(a) or a >= 0:
        return {"slope": float(a), "rul_days": float("inf"), "pred_end_date": None}

    # 너무 완만한 감소 → 보수화(더 큰 감속 가정)
    a_eff = min(a, slope_floor)  # 둘 다 음수: 더 작은(더 음수) 값 선택
    delta = max(h_now - health_end, 0.0)
    if delta <= 0:
        # 이미 임계치 이하
        return {"slope": float(a), "rul_days": 0.0, "pred_end_date": last_date}

    rul = float(delta / (-a_eff))
    if max_rul_days is not None:
        rul = min(rul, float(max_rul_days))

    return {
        "slope": float(a),
        "rul_days": float(rul),
        "pred_end_date": last_date + pd.Timedelta(days=rul),
    }

# =========================
# 3) 하루 샘플 → MSE/Health 계산
# =========================
def sample_health_from_run(run_df: pd.DataFrame, mapper, mse_col: str = "recon_error_mse", agg: str = "mean") -> tuple:
    """
    run_df: 해당 '하루 샘플'(2048행 등). mse_col에 per-row mse가 있다고 가정.
    agg: 'mean' | 'median' | 'p90'
    """
    vals = pd.to_numeric(run_df[mse_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        raise ValueError("run_df에 유효한 MSE가 없습니다.")
    if agg == "mean":
        run_mse = float(vals.mean())
    elif agg == "median":
        run_mse = float(vals.median())
    elif agg.lower() in ("p90", "p95"):
        q = 0.9 if agg.lower() == "p90" else 0.95
        run_mse = float(vals.quantile(q))
    else:
        raise ValueError(f"지원하지 않는 agg: {agg}")

    h = float(mapper(run_mse))
    h = float(np.clip(h, 0.0, 100.0))
    return run_mse, h

# =========================
# 4) 최종 헬퍼: (샘플, 모드) → MSE/Health/Slope/RUL
# =========================
def evaluate_sample_with_mode(
        run_df: pd.DataFrame,                              # 하루 샘플 (열에 'recon_error_mse' 포함)
        sample_date: Union[str, pd.Timestamp],             # 이 샘플의 날짜
        mode: int,                                         # 1/2/3
        daily_by_mode: Dict[int, pd.DataFrame],            # {1: daily_df_m1, 2: daily_df_m2, 3: daily_df_m3}
        scaleinfo_by_mode: Dict[int, Dict],                # {1: scale_info_m1, ...}  (lo/hi/gamma 등)
        agg: str = "mean",
        window_days: int = 30,
        health_end: float = 20.0,
        slope_floor: float = -0.2,
        max_rul_days: float = 365.0
) -> Dict[str, object]:
    """
    반환: dict(run_mse, health_now, slope, rul_days, pred_end_date)
    """
    if mode not in scaleinfo_by_mode:
        raise KeyError(f"scaleinfo_by_mode에 mode={mode} 키가 없습니다.")
    if mode not in daily_by_mode:
        raise KeyError(f"daily_by_mode에 mode={mode} 키가 없습니다.")

    # 1) 헬스 매퍼 준비(모드별 scale info)
    mapper = build_health_mapper_from_scaleinfo(scaleinfo_by_mode[mode])

    # 2) 샘플 → MSE, health_now
    run_mse, health_now = sample_health_from_run(run_df, mapper, mse_col="recon_error_mse", agg=agg)

    # 3) 모드별 일일 DF에서 slope/RUL
    daily_df = daily_by_mode[mode].copy()  # ['date','recon_error_mse','health'] 가정
    out = compute_rul_days_refined(
        daily_df=daily_df,
        window_days=window_days,
        health_now=health_now,
        asof_date=sample_date,
        health_col="health",
        health_end=health_end,
        slope_floor=slope_floor,
        max_rul_days=max_rul_days
    )

    return {
        "run_mse": run_mse,
        "health_now": health_now,
        "slope_per_day": out["slope"],
        "rul_days": out["rul_days"],
        "pred_end_date": out["pred_end_date"],
    }

# =========================
# 5) 사용 예시 (이미 계산된 모드별 DF/스케일이 있다고 가정)
# =========================
data1 = {
    "date": [
        "2025-06-19","2025-06-20","2025-06-29","2025-07-05","2025-07-06",
        "2025-07-10","2025-07-11","2025-07-12","2025-07-24","2025-08-02",
        "2025-08-08","2025-08-15","2025-08-18","2025-08-21","2025-08-25",
        "2025-08-28","2025-08-30","2025-09-12","2025-09-19","2025-10-02",
        "2025-10-23","2025-10-24","2025-10-26","2025-10-30","2025-11-07",
        "2025-11-22","2025-11-28","2025-12-10","2025-12-25"
    ],
    "recon_error_mse": [
        0.000052,0.000250,0.000065,0.000069,0.000076,
        0.000083,0.000082,0.000177,0.000084,0.000077,
        0.000076,0.000065,0.000078,0.000070,0.000102,
        0.000097,0.000086,0.000098,0.000156,0.000430,
        0.000336,0.000385,0.000280,0.000217,0.000421,
        0.000370,0.000433,0.000531,0.007167
    ],
    "health": [
        94.643574,56.450304,92.415201,91.619867,90.399939,
        89.211968,89.361808,71.623099,89.073220,90.208367,
        90.375062,92.348882,90.032564,91.582050,85.789982,
        86.696808,88.637670,86.405871,75.659410,1.526659,
        35.919567,21.702206,49.827022,63.511327,7.530760,
        26.360856,0.000000,0.000000,0.000000
    ]
}

data2 = {
    "date": [
        "2025-06-22","2025-07-04","2025-08-03","2025-08-08","2025-08-15",
        "2025-08-17","2025-08-22","2025-08-24","2025-08-29","2025-10-24",
        "2025-10-31","2025-11-08","2025-12-18","2025-12-28"
    ],
    "recon_error_mse": [
        0.000055,0.000021,0.000022,0.000019,0.000024,
        0.000021,0.000023,0.000020,0.000023,0.000051,
        0.000062,0.000076,0.000047,0.000270
    ],
    "health": [
        94.182784,100.000000,99.816374,100.000000,99.532882,
        99.991331,99.735925,100.000000,99.698907,94.930990,
        92.970413,90.373872,95.639187,52.021910
    ]
}

data3 = {
    "date": [
        "2025-01-22","2025-01-26","2025-01-31","2025-02-23","2025-03-22",
        "2025-05-16","2025-05-22","2025-08-11","2025-08-19"
    ],
    "recon_error_mse": [
        0.000074,0.000112,0.000094,0.000238,0.000072,
        0.000144,0.000196,0.000073,0.000087
    ],
    "health": [
        90.742485,83.924655,87.135742,59.065628,91.159643,
        77.907903,67.721872,90.971750,88.509374
    ]
}

daily_m1 = pd.DataFrame(data1)
daily_m2 = pd.DataFrame(data2)
daily_m3 = pd.DataFrame(data3)
scale_info_m1 = {'p_low': 0.05, 'p_high': 0.95, 'gamma': 0.7, 'use_log': True, 'lo': 2.1239315679408013e-05, 'hi': 0.00043123801497343684}
scale_info_m2 = {'p_low': 0.05, 'p_high': 0.95, 'gamma': 0.7, 'use_log': True, 'lo': 2.1239315679408013e-05, 'hi': 0.00043123801497343684}
scale_info_m3 = {'p_low': 0.05, 'p_high': 0.95, 'gamma': 0.7, 'use_log': True, 'lo': 2.1239315679408013e-05, 'hi': 0.00043123801497343684}
daily_by_mode = {1: daily_m1, 2: daily_m2, 3: daily_m3}
scaleinfo_by_mode = {1: scale_info_m1, 2: scale_info_m2, 3: scale_info_m3}
'''
sample_run_df = results_s6
result = evaluate_sample_with_mode(
    run_df=sample_run_df,
    sample_date="2025-10-24",
    mode=3,
    daily_by_mode=daily_by_mode,
    scaleinfo_by_mode=scaleinfo_by_mode,
    agg="mean",
    window_days=30,
    health_end=20.0,
    slope_floor=-0.2,
    max_rul_days=365.0)
'''

def compute_rul_for_sample(mode: int, blade_id: int) -> dict:
    daily_m1 = pd.DataFrame({
        "date": [
            "2025-06-19","2025-06-20","2025-06-29","2025-07-05","2025-07-06",
            "2025-07-10","2025-07-11","2025-07-12","2025-07-24","2025-08-02",
            "2025-08-08","2025-08-15","2025-08-18","2025-08-21","2025-08-25",
            "2025-08-28","2025-08-30","2025-09-12","2025-09-19","2025-10-02",
            "2025-10-23","2025-10-24","2025-10-26","2025-10-30","2025-11-07",
            "2025-11-22","2025-11-28","2025-12-10","2025-12-25"
        ],
        "recon_error_mse": [
            0.000052,0.000250,0.000065,0.000069,0.000076,
            0.000083,0.000082,0.000177,0.000084,0.000077,
            0.000076,0.000065,0.000078,0.000070,0.000102,
            0.000097,0.000086,0.000098,0.000156,0.000430,
            0.000336,0.000385,0.000280,0.000217,0.000421,
            0.000370,0.000433,0.000531,0.007167
        ],
        "health": [
            94.643574,56.450304,92.415201,91.619867,90.399939,
            89.211968,89.361808,71.623099,89.073220,90.208367,
            90.375062,92.348882,90.032564,91.582050,85.789982,
            86.696808,88.637670,86.405871,75.659410,1.526659,
            35.919567,21.702206,49.827022,63.511327,7.530760,
            26.360856,0.000000,0.000000,0.000000
        ]
    })
    daily_m2 = pd.DataFrame({
        "date": [
            "2025-06-22","2025-07-04","2025-08-03","2025-08-08","2025-08-15",
            "2025-08-17","2025-08-22","2025-08-24","2025-08-29","2025-10-24",
            "2025-10-31","2025-11-08","2025-12-18","2025-12-28"
        ],
        "recon_error_mse": [
            0.000055,0.000021,0.000022,0.000019,0.000024,
            0.000021,0.000023,0.000020,0.000023,0.000051,
            0.000062,0.000076,0.000047,0.000270
        ],
        "health": [
            94.182784,100.000000,99.816374,100.000000,99.532882,
            99.991331,99.735925,100.000000,99.698907,94.930990,
            92.970413,90.373872,95.639187,52.021910
        ]
    })
    daily_m3 = pd.DataFrame({
        "date": [
            "2025-01-22","2025-01-26","2025-01-31","2025-02-23","2025-03-22",
            "2025-05-16","2025-05-22","2025-08-11","2025-08-19"
        ],
        "recon_error_mse": [
            0.000074,0.000112,0.000094,0.000238,0.000072,
            0.000144,0.000196,0.000073,0.000087
        ],
        "health": [
            90.742485,83.924655,87.135742,59.065628,91.159643,
            77.907903,67.721872,90.971750,88.509374
        ]
    })
    scale_info_m1 = {'p_low': 0.05, 'p_high': 0.95, 'gamma': 0.7, 'use_log': True, 'lo': 2.1239315679408013e-05, 'hi': 0.00043123801497343684}
    scale_info_m2 = {'p_low': 0.05, 'p_high': 0.95, 'gamma': 0.7, 'use_log': True, 'lo': 2.1239315679408013e-05, 'hi': 0.00043123801497343684}
    scale_info_m3 = {'p_low': 0.05, 'p_high': 0.95, 'gamma': 0.7, 'use_log': True, 'lo': 2.1239315679408013e-05, 'hi': 0.00043123801497343684}
    daily_by_mode = {1: daily_m1, 2: daily_m2, 3: daily_m3}
    scaleinfo_by_mode = {1: scale_info_m1, 2: scale_info_m2, 3: scale_info_m3}

    key = (int(mode), int(blade_id))
    if key not in SAMPLE_MAP:
        raise ValueError(f"No sample for mode={mode}, blade_id={blade_id}")

    results_df, df_raw = SAMPLE_MAP[key]
    print(results_df.head())

    # 1) 샘플 꺼내기
    result = evaluate_sample_with_mode(
        run_df=results_df,
        sample_date="2025-09-12",
        mode=mode,
        daily_by_mode=daily_by_mode,
        scaleinfo_by_mode=scaleinfo_by_mode,
        agg="mean",
        window_days=30,
        health_end=20.0,
        slope_floor=-0.2,
        max_rul_days=365.0)
    pred = result.get("pred_end_date")
    pred_str = pd.to_datetime(pred).strftime("%Y-%m-%d") if pred is not None else None

    return {
        "health_now": float(result.get("health_now")) if result.get("health_now") is not None else None,
        "rul_days": (float(result.get("rul_days"))
                     if (result.get("rul_days") not in (None, float("inf")))
                     else None),
        "pred_end_date": pred_str,
    }
results = compute_rul_for_sample(1,224)
print(results)