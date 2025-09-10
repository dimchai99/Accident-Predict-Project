# app/model_runner.py
from typing import Optional
import os, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


BASE = Path(__file__).resolve().parents[1]

# .env가 없어도 안전한 기본값(절대경로)
MODEL_PATH  = os.getenv("MODEL_PATH",  str(BASE / "models" / "fcast.h5"))
SCALER_PATH = os.getenv("SCALER_PATH", str(BASE / "models" / "scaler.pkl"))

# ── 아티팩트 로드 (lazy)
_model  = None
_scaler = None
_ecdf_ref = None
_calib = None

def _lazy_load():
    global _model, _scaler, _ecdf_ref, _calib
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _scaler is None:
        import joblib
        _scaler = joblib.load(SCALER_PATH)
    try:
        with open(BASE / "models" / "calib.json", "r") as f:
            _calib = json.load(f)
        _ecdf_ref = np.load(BASE / "models" / "ecdf_ref.npy")
    except Exception:
        _calib = {"H_MAX":95.0,"H_MIN":5.0,"TAU":0.007,"p0":0.5,"b":0.0,"gamma":1.0}
        _ecdf_ref = None

def _build_ecdf_smooth(x, alpha=0.5, beta=0.5):
    xs = np.sort(np.asarray(x)); n = xs.size
    def F(val):
        r = np.searchsorted(xs, val, side="right")
        return (r + alpha) / (n + alpha + beta)
    return F

def _health_from_mse(mse_arr: np.ndarray):
    H_MAX, H_MIN = _calib["H_MAX"], _calib["H_MIN"]
    TAU, p0, b, gamma = _calib["TAU"], _calib["p0"], _calib["b"], _calib["gamma"]
    if _ecdf_ref is not None:
        F_new = _build_ecdf_smooth(_ecdf_ref, 0.5, 0.5)
        p_series = np.array([F_new(v) for v in mse_arr])
    else:
        mn, mx = float(np.min(mse_arr)), float(np.max(mse_arr)+1e-9)
        p_series = (mse_arr - mn) / max(mx - mn, 1e-9)
    p_shift  = 1.0 / (1.0 + np.exp(-(((p_series - p0)/max(1e-9,TAU)) + b)))
    health   = H_MIN + (H_MAX - H_MIN) * np.power((1.0 - p_shift), gamma)
    return health

def infer_df(df: pd.DataFrame, window_size=128, stride=1):
    """
    입력: blade_benchmark에서 읽어온 DF (relative_timestamp, cut_*, film_*)
    출력: [{timestamp, mse, health}, ...]  (timestamp = 윈도우 마지막 시점)
    """
    _lazy_load()

    keep_cols = [
        "cut_torque","cut_lag_error","cut_position","cut_speed",
        "film_position","film_speed","film_lag_error"
    ]
    X2d = _scaler.transform(df[keep_cols].values)
    ts_all = df["relative_timestamp"].to_numpy()

    N, p = X2d.shape
    win = window_size
    out_ts, mse_list = [], []

    for s in range(0, N - win, stride):
        X_win  = X2d[s:s+win]              # (win, p)
        y_true = X2d[s+win:s+win+1]        # 다음 1스텝 (1, p)
        if len(y_true) == 0:
            break
        y_hat = _model.predict(X_win[None, ...], verbose=0)  # (1, p)
        mse = float(np.mean((y_hat[0] - y_true[0])**2))
        mse_list.append(mse)
        out_ts.append(float(ts_all[s+win]))  # ✅ 윈도우 마지막 시점

    if not mse_list:
        return []

    mse_arr = np.array(mse_list, dtype=float)
    health  = _health_from_mse(mse_arr)

    return [
        {"timestamp": out_ts[i], "mse": float(mse_arr[i]), "health": float(health[i])}
        for i in range(len(mse_arr))
    ]


# 💡 그래프/학습에 쓸 컬럼: DB/DF에 실제 존재하는 이름으로 맞추세요.
KEEP_COLS = [
    "cut_torque", "cut_lag_error", "cut_speed",
    "film_speed", "film_lag_error",
    # 필요 시 포함: "cut_position", "film_position"
]

WIN  = 128   # window size
STEP = 1     # stride

def _ensure_model():
    """내부에서 모델/스케일러 lazy-load (기존 _lazy_load() 사용)"""
    _lazy_load()
    return _model  # fcast 같은 전역 모델

def _pred_errors(model, X, y, metric="mse"):
    """
    LSTM(seq2one) 기준: X.shape=(N, win, p), y.shape=(N, p)
    모델의 next-step 예측 y_hat과의 오차를 계산.
    """
    y_hat = model.predict(X, verbose=0)
    if metric == "mse":
        e = np.mean((y_hat - y) ** 2, axis=1)
    else:
        raise ValueError("Only 'mse' supported")
    return e, y_hat

def _df_to_seq2one(df, win=WIN, step=STEP, keep_cols=KEEP_COLS):
    X2d = df[keep_cols].to_numpy()
    N, p = X2d.shape
    Xs, Ys, idx = [], [], []
    last = N - win
    for s in range(0, last, step):
        Xs.append(X2d[s:s+win])
        Ys.append(X2d[s+win])
        idx.append(s+win)
    if not Xs:
        return np.empty((0, win, p)), np.empty((0, p)), np.array([])
    return np.stack(Xs), np.stack(Ys), np.array(idx)

def mse_series_from_df(df, win=WIN, step=STEP):
    """
    입력 df → (x축, MSE시퀀스)
    x축: timestamp가 있으면 '윈도우 끝 timestamp - 시작 timestamp', 없으면 0..N-1
    """
    model = _ensure_model()

    # 스케일링/전처리 루틴이 이미 _scaler 등에 있으면 적용
    X_df = df.copy()
    if _scaler is not None:
        arr = _scaler.transform(X_df[KEEP_COLS].values)
        X_df[KEEP_COLS] = arr

    X, y, idx = _df_to_seq2one(X_df, win=win, step=step)
    if len(idx) == 0:
        return np.array([]), np.array([])

    e, _ = _pred_errors(model, X, y, metric="mse")
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy()
        x = ts[idx] - ts[0]
    else:
        x = np.arange(len(e))
    return x, e

# ====== Health 계산(ECDF 기반) ======

USE_LOG = False  # 필요 시 raw MSE 대신 log1p 사용

def file_score_series_LSTM(df, win=WIN, step=STEP):
    """(윈도우 끝 timestamp 배열, (log)MSE score 배열)"""
    model = _ensure_model()

    # 스케일링
    X_df = df.copy()
    if _scaler is not None:
        X_df[KEEP_COLS] = _scaler.transform(X_df[KEEP_COLS].values)

    X, y, idx = _df_to_seq2one(X_df, win=win, step=step)
    if len(idx) == 0:
        return np.array([]), np.array([])

    e, _ = _pred_errors(model, X, y, metric="mse")
    s = np.log1p(e) if USE_LOG else e
    ts = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.arange(len(df))
    return ts[idx], s

def build_ecdf_smooth(x, alpha=0.5, beta=0.5):
    xs = np.sort(np.asarray(x)); n = xs.size
    def F(val):
        r = np.searchsorted(xs, val, side="right")
        return (r + alpha) / (n + alpha + beta)
    return F

# 🔹 NEW 참조 분포가 미리 아티팩트(_ecdf_ref)로 로드돼 있으면 우선 사용
#    없으면 런타임에 on-the-fly로 해당 df 기반 fallback 가능
def get_ref_ecdf(scores_new_ref: Optional[np.ndarray]):
    if _ecdf_ref is not None:               # (선행 학습 시 저장해둔 참조)
        return _ecdf_ref
    if scores_new_ref is not None:
        return build_ecdf_smooth(scores_new_ref, alpha=0.5, beta=0.5)
    # 정말 없으면 단조 증가 함수(퀵&더티)
    return build_ecdf_smooth(np.array([0.0, 1.0]), alpha=0.5, beta=0.5)

def health_ecdf_soft_calib_for_df(
        df,
        F_new=None,
        agg_p=95,
        last_seconds=None,
        # 캘리브레이션 파라미터 (아티팩트에 있으면 _calib로부터 로드)
        H_MAX=95.0, H_MIN=5.0, H_new=95.0, H_wrn=15.0, TAU=0.007,
        p0=None, p_w_med=None, b=None, gamma=None,
):
    """
    df 한 파일(=하나 prefix)에 대해 Health 점 하나 산출:
    - file_score_series_LSTM → 점수시퀀스 s
    - 구간(percentile) agg 후 NEW ECDF로 분위 p 계산
    - p를 시그모이드/지수 매핑하여 최종 health(%) 반환
    """
    ts, s = file_score_series_LSTM(df)
    if len(s) == 0:
        return dict(health=np.nan, agg=np.nan, p=np.nan, p_shift=np.nan)

    # 집계 구간
    if (last_seconds is None) or ((ts[-1] - ts[0]) < float(last_seconds)):
        agg = np.percentile(s, agg_p)
    else:
        m = ts >= (ts[-1] - last_seconds)
        if m.sum() == 0: m[:] = True
        agg = np.percentile(s[m], agg_p)

    # 참조 ECDF
    if F_new is None:
        F_new = get_ref_ecdf(scores_new_ref=None)  # 필요 시 외부에서 주입 가능
    p = float(F_new(agg))

    # 캘리브레이션 파라미터 (_calib에 있으면 우선 적용)
    if _calib:
        H_MAX = _calib.get("H_MAX", H_MAX)
        H_MIN = _calib.get("H_MIN", H_MIN)
        H_new = _calib.get("H_new", H_new)
        H_wrn = _calib.get("H_wrn", H_wrn)
        TAU   = _calib.get("TAU", TAU)
        p0    = _calib.get("p0", p0)
        p_w_med = _calib.get("p_w_med", p_w_med)
        b     = _calib.get("b", b)
        gamma = _calib.get("gamma", gamma)

    # p0, b, gamma 가 미정이면 간단 계산(러프한 fallback)
    def logit(u): return np.log(u/(1-u))
    if p0 is None:      p0 = 0.5
    s0 = 1.0 - ((H_new - H_MIN) / (H_MAX - H_MIN))
    if b  is None:      b  = logit(np.clip(s0, 1e-6, 1-1e-6))
    if p_w_med is None: p_w_med = 0.9
    if gamma is None:
        s_w = 1.0 / (1.0 + np.exp(-(((p_w_med - p0)/max(1e-9,TAU)) + b)))
        target = (H_wrn - H_MIN) / (H_MAX - H_MIN)
        gamma  = np.log(np.clip(target,1e-6,1-1e-6)) / np.log(np.clip(1.0 - s_w,1e-9,1.0))

    # 시그모이드 매핑
    z = (p - p0)/max(1e-9, TAU) + b
    p_shift = 1.0/(1.0 + np.exp(-z))
    health  = H_MIN + (H_MAX - H_MIN) * ((1.0 - p_shift) ** gamma)

    return dict(
        health=float(np.clip(health, H_MIN, H_MAX)),
        agg=float(agg), p=p, p_shift=float(p_shift)
    )