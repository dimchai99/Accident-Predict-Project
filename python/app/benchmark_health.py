# app/benchmark_health.py
import sys, traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import json
# app.db 연결 재사용
ROOT = Path(__file__).resolve().parents[1]  # .../Accident-Predict-Project/python
sys.path.insert(0, str(ROOT))
from app.db import get_cursor
import os
import argparse

# ==== 설정 ====
# NEW / WORN 구분 (필요시 바꿔줘)
NEW_PREFIXES  = ["a", "b", "c"]
WORN_PREFIXES = ["d", "e", "f"]

USE_LOG       = False   # MSE 대신 log1p(MSE)를 점수로 쓰고 싶으면 True
AGG_P         = 95      # 퍼센타일 집계 (단일 스코어)
LAST_SECONDS: Optional[float] = None  # 최근 n초만 사용할지 (None이면 전체 사용)

H_MAX, H_MIN  = 95.0, 5.0  # 최종 Health 상/하한
H_new         = 95.0       # NEW 중앙에 매핑될 목표 Health(%)
H_wrn         = 15.0       # WORN 중앙에 매핑될 목표 Health(%)
TAU           = 0.007      # p0 주변 부드러움

# ==== 유틸 ====
def build_ecdf_smooth(x: np.ndarray, alpha: float = 0.5, beta: float = 0.5):
    xs = np.sort(np.asarray(x, dtype=float)); n = xs.size
    def F(val: float) -> float:
        r = np.searchsorted(xs, val, side="right")
        return (r + alpha) / (n + alpha + beta)
    return F

def load_mse_series_by_prefix(prefix: str) -> pd.DataFrame:
    """benchmark_mse에서 prefix(첫 글자)별 time_stamp, mse 로드"""
    with get_cursor() as cur:
        cur.execute("""
                    SELECT time_stamp, mse, blade_inbound_id
                    FROM benchmark_mse
                    WHERE LEFT(blade_inbound_id, 1) = %s
                    ORDER BY time_stamp ASC
                    """, (prefix,))
        rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["time_stamp","mse","blade_inbound_id"])

def series_window(ts: np.ndarray, x: np.ndarray, last_seconds: Optional[float]) -> np.ndarray:
    """최근 last_seconds 구간만 마스크(없으면 전체 True)"""
    if last_seconds is None or len(ts) == 0:
        return np.ones_like(x, dtype=bool)
    t1 = float(ts[-1])
    mask = ts >= (t1 - float(last_seconds))
    if not mask.any():
        mask = np.ones_like(x, dtype=bool)
    return mask

def series_score(mse: np.ndarray) -> np.ndarray:
    """점수 스케일 선택 (raw MSE 또는 log1p)"""
    return np.log1p(mse) if USE_LOG else mse

def agg_percentile(ts: np.ndarray, mse: np.ndarray, agg_p: int, last_seconds: Optional[float]) -> Optional[float]:
    """한 prefix 시리즈에서 단일 집계 스코어(퍼센타일) 계산"""
    if len(mse) == 0:
        return None
    mask = series_window(ts, mse, last_seconds)
    s = series_score(mse[mask])
    if s.size == 0:
        return None
    return float(np.percentile(s, agg_p))

# ==== 메인 로직: DB에서 참조분포/보정/헬스 산출 ====
def build_reference_from_new() -> Tuple[np.ndarray, Dict[str, float]]:
    """NEW prefix들의 점수 시계열을 합쳐 참조분포(ECDF)와 보정 파라미터 산출에 필요한 통계 생성"""
    agg_list_new = []
    scores_new_concat = []

    # NEW: 집계값(agg)도 모으고, 참조분포용 시퀀스도 수집
    for px in NEW_PREFIXES:
        df = load_mse_series_by_prefix(px)
        if df.empty:
            continue
        ts = df["time_stamp"].to_numpy(dtype=float)
        mse = df["mse"].to_numpy(dtype=float)
        # 참조분포: 전체(또는 최근구간) 점수를 concat
        mask = series_window(ts, mse, LAST_SECONDS)
        s_seq = series_score(mse[mask])
        if s_seq.size:
            scores_new_concat.append(s_seq)
        # 집계값
        agg = agg_percentile(ts, mse, AGG_P, LAST_SECONDS)
        if agg is not None:
            agg_list_new.append(agg)

    if not scores_new_concat:
        # NEW가 비어도 로직이 돌아가게 방어
        scores_new_ref = np.array([0.0], dtype=float)
    else:
        scores_new_ref = np.concatenate(scores_new_concat).astype(float)

    p0 = float(np.median(agg_list_new)) if agg_list_new else 0.5  # NEW 중앙 분위에 대응시킬 '점수 공간의 중앙값'

    # ECDF는 '점수값 -> 분위' 변환이므로, 보정 파라미터 산출엔 분위가 아니라 집계 '값'이 들어옴
    # 이 p0는 이후 'ECDF(agg)'의 p와 비교될 기준이므로 수치적으로는 실제 중앙점의 점수값 개념.

    stats = {"p0_score_space": p0}  # 참고 정보
    return scores_new_ref, stats


def _parse_target_prefix() -> Optional[str]:
    """
    우선순위: CLI 인자(예: `python -m app.benchmark_health a`) > 환경변수 HEALTH_PREFIX
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("prefix", nargs="?", help="단일 접두사 문자 (예: a)")
    args, _ = p.parse_known_args()
    px = args.prefix or os.getenv("HEALTH_PREFIX")
    if px:
        px = px.strip()
        if len(px) != 1:
            print(f"❌ prefix는 한 글자여야 합니다. 받은 값: '{px}'")
            return None
        return px.lower()
    return None

def compute_health_single(prefix: str, F_new, calib) -> Optional[Dict[str, float]]:
    """prefix 하나의 Health 단일값 산출"""
    df = load_mse_series_by_prefix(prefix)
    if df.empty:
        return None
    ts  = df["time_stamp"].to_numpy(dtype=float)
    mse = df["mse"].to_numpy(dtype=float)

    # 1) 단일 집계 스코어
    agg = agg_percentile(ts, mse, AGG_P, LAST_SECONDS)
    if agg is None:
        return None

    # 2) 참조 ECDF로 분위 p 계산
    p = float(F_new(agg))

    # 3) 시그모이드 보정 + 감마 맵핑 → Health
    H_MAX, H_MIN = calib["H_MAX"], calib["H_MIN"]
    TAU, p0, b, gamma = calib["TAU"], calib["p0"], calib["b"], calib["gamma"]

    z = ((p - p0) / max(1e-9, TAU)) + b
    p_shift = 1.0 / (1.0 + np.exp(-z))
    health  = H_MIN + (H_MAX - H_MIN) * ((1.0 - p_shift) ** gamma)
    health  = float(np.clip(health, H_MIN, H_MAX))

    return {"health": health, "agg": float(agg), "p": p, "p_shift": float(p_shift)}

ART_ECDF  = "models/ecdf_ref.npy"
ART_CALIB = "models/calib.json"

def try_load_artifacts():
    """저장된 ECDF/캘리브레이션 파라미터 로드. 둘 다 있으면 (F_new, calib) 반환, 없으면 None."""
    import json, numpy as np, os
    if not (os.path.exists(ART_ECDF) and os.path.exists(ART_CALIB)):
        return None
    ECDF_REF = np.load(ART_ECDF)
    with open(ART_CALIB, "r", encoding="utf-8") as f:
        CALIB = json.load(f)
    F_new = build_ecdf_smooth(ECDF_REF, alpha=0.5, beta=0.5)
    return F_new, CALIB

def main():
    # 0) 저장본 로드 시도
    loaded = try_load_artifacts()
    if loaded is not None:
        F_new, CALIB = loaded
        calib = dict(
            H_MAX=CALIB["H_MAX"], H_MIN=CALIB["H_MIN"],
            TAU=CALIB["TAU"], p0=CALIB["p0"], b=CALIB["b"], gamma=CALIB["gamma"],
        )
        print("[info] loaded artifacts -> ecdf_ref.npy + calib.json")
        print(f"[calib] p0={calib['p0']:.4f}, b={calib['b']:.3f}, gamma={calib['gamma']:.3f}")

    else:
        # --- 최초 1회 계산 & 저장 ---
        scores_new_ref, _ = build_reference_from_new()
        F_new = build_ecdf_smooth(scores_new_ref, alpha=0.5, beta=0.5)

        def agg_list(prefixes: List[str]) -> List[float]:
            out = []
            for px in prefixes:
                df = load_mse_series_by_prefix(px)
                if df.empty:
                    continue
                ts  = df["time_stamp"].to_numpy(dtype=float)
                mse = df["mse"].to_numpy(dtype=float)
                a = agg_percentile(ts, mse, AGG_P, LAST_SECONDS)
                if a is not None:
                    out.append(a)
            return out

        aggs_new  = agg_list(NEW_PREFIXES)
        aggs_worn = agg_list(WORN_PREFIXES)

        p_new = np.array([F_new(a) for a in aggs_new],  dtype=float) if aggs_new  else np.array([0.5])
        p_wrn = np.array([F_new(a) for a in aggs_worn], dtype=float) if aggs_worn else np.array([0.5])

        p0      = float(np.median(p_new))
        p_w_med = float(np.median(p_wrn))

        def logit(u: float) -> float:
            u = float(np.clip(u, 1e-6, 1 - 1e-6))
            return float(np.log(u/(1.0 - u)))

        s0 = 1.0 - ((H_new - H_MIN) / (H_MAX - H_MIN))
        b  = logit(s0)
        s_w = 1.0 / (1.0 + np.exp(-(((p_w_med - p0)/max(1e-9, TAU)) + b)))
        target = (H_wrn - H_MIN) / (H_MAX - H_MIN)
        gamma  = float(np.log(np.clip(target,1e-6,1-1e-6)) / np.log(np.clip(1.0 - s_w,1e-9,1.0)))

        # 저장 (주의: 네 출력엔 models/ 경로가 쓰였으니 그걸 유지)
        np.save(ART_ECDF, scores_new_ref)
        with open(ART_CALIB, "w", encoding="utf-8") as f:
            json.dump({"H_MAX": H_MAX, "H_MIN": H_MIN,
                       "TAU": TAU, "p0": p0, "b": b, "gamma": gamma}, f)
        print(f"[calib] saved -> {ART_ECDF}, {ART_CALIB}")

        calib = dict(H_MAX=H_MAX, H_MIN=H_MIN, TAU=TAU, p0=p0, b=b, gamma=gamma)
        print(f"[calib] p0={p0:.4f}, p_w_med={p_w_med:.4f}, s0={s0:.3f}, b={b:.3f}, gamma={gamma:.3f}")

    # === 공통 출력: 여기로 이동 ===
    # === 단일 prefix만 계산/출력 ===
    target = _parse_target_prefix()
    if not target:
        print("사용법: python -m app.benchmark_health <prefix>")
        print("예시  : python -m app.benchmark_health a")
        print("또는  : HEALTH_PREFIX=a python -m app.benchmark_health")
        return

    print(f"[info] TARGET prefix='{target}', AGG_P={AGG_P}, LAST_SECONDS={LAST_SECONDS}, USE_LOG={USE_LOG}")
    res = compute_health_single(target, F_new, calib)
    if res is None:
        print(f"{target}: no data (benchmark_mse에 해당 prefix 데이터 없음)")
    else:
        print(f"{target}: HEALTH={res['health']:.2f}%  (agg={res['agg']:.6f}, p={res['p']:.4f}, p_shift={res['p_shift']:.4f})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ ERROR:", e)
        traceback.print_exc()
        sys.exit(2)
