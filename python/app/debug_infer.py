# app/debug_infer.py  — infer_df 경로 점검(콘솔 출력 전용)
import os, json, traceback
import numpy as np
import pandas as pd

# 3.9 호환
from typing import Optional

from app.db import get_cursor
from app.model_runner import infer_df

# (선택) 내부 아티팩트 확인용
from app.model_runner import _lazy_load, MODEL_PATH, SCALER_PATH
try:
    from app.model_runner import _model, _scaler
except Exception:
    _model = None
    _scaler = None

def main(prefix: str = "a", limit: int = 300, win: int = 128, step: int = 1):
    print("=== ARTIFACT PATHS ===")
    print("MODEL_PATH :", MODEL_PATH, "exists?", os.path.exists(MODEL_PATH))
    print("SCALER_PATH:", SCALER_PATH, "exists?", os.path.exists(SCALER_PATH))

    # 데이터 가져오기
    with get_cursor() as cur:
        cur.execute("""
                    SELECT *
                    FROM blade_benchmark
                    WHERE blade_benchmark_id LIKE %s
                    ORDER BY relative_timestamp ASC, blade_benchmark_id ASC
                        LIMIT %s
                    """, (prefix + '%', int(limit)))
        rows = cur.fetchall()

    print(f"\n=== FETCH === prefix='{prefix}' rows={len(rows)}")
    if not rows:
        print("No rows; 다른 prefix로 다시 시도해보세요.")
        return

    df = pd.DataFrame(rows)
    print("columns:", list(df.columns))
    need = {
        "relative_timestamp",
        "cut_torque","cut_lag_error","cut_position","cut_speed",
        "film_position","film_speed","film_lag_error"
    }
    missing = [c for c in need if c not in df.columns]
    if missing:
        print("❌ Missing columns:", missing)
        return

    # 아티팩트 로드 상태/입출력 차원 확인
    try:
        _lazy_load()
        try:
            in_shape  = getattr(_model, "input_shape", None)
            out_shape = getattr(_model, "output_shape", None)
        except Exception:
            in_shape = out_shape = None

        sc_mean = getattr(_scaler, "mean_", None)
        print("\n=== MODEL/SCALER ===")
        print("model.input_shape :", in_shape)
        print("model.output_shape:", out_shape)
        if sc_mean is not None:
            print("scaler.mean_.shape:", np.shape(sc_mean))
        else:
            print("scaler.mean_     :", None)
    except Exception as e:
        print("\n❌ ERROR loading artifacts:", type(e).__name__, e)
        traceback.print_exc()
        return

    # 실제 infer_df 실행 (가장 중요한 부분)
    try:
        print(f"\n=== INFER (win={win}, step={step}) ===")
        results = infer_df(df, window_size=win, stride=step)
        print("len(results):", len(results))
        if results:
            print("head:", json.dumps(results[:3], ensure_ascii=False, indent=2))
            print("tail:", json.dumps(results[-3:], ensure_ascii=False, indent=2))
        else:
            print("⚠️ 결과가 비었습니다. (윈도우 부족이거나, 내부 로직 조건 미충족)")
    except Exception as e:
        print("\n❌ ERROR in infer_df:", type(e).__name__, e)
        traceback.print_exc()
        return

if __name__ == "__main__":
    # 필요시 여기 값 바꿔서 테스트
    main(prefix="a", limit=300, win=128, step=1)
