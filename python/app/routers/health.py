# app/routers/health.py
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict
from app.benchmark_health import (
    try_load_artifacts,
    build_reference_from_new,
    build_ecdf_smooth,
    compute_health_single,
    USE_LOG, AGG_P, LAST_SECONDS,
)

router = APIRouter(prefix="/health", tags=["health"])

# lazy 초기화 (ECDF / 캘리브레이션)
_F_new = None
_calib = None

def _ensure_ready():
    global _F_new, _calib
    if _F_new is not None and _calib is not None:
        return
    loaded = try_load_artifacts()
    if loaded is not None:
        F_new, CALIB = loaded
        _F_new = F_new
        _calib = dict(
            H_MAX=CALIB["H_MAX"], H_MIN=CALIB["H_MIN"],
            TAU=CALIB["TAU"], p0=CALIB["p0"], b=CALIB["b"], gamma=CALIB["gamma"],
        )
    else:
        # 저장본이 없으면 NEW 분포로 즉석 ECDF 구성 (안전한 기본치)
        scores_new_ref, _ = build_reference_from_new()
        _F_new = build_ecdf_smooth(scores_new_ref, alpha=0.5, beta=0.5)
        _calib = dict(H_MAX=95.0, H_MIN=5.0, TAU=0.007, p0=0.5, b=0.0, gamma=1.0)

@router.get("/by_prefix/{prefix}")
def get_health_by_prefix(prefix: str) -> Dict:
    """
    프런트에서 사용하는 엔드포인트:
    GET /health/by_prefix/{prefix}
    응답: { prefix, result: {health, agg, p, p_shift} | None, meta: {...} }
    """
    if not prefix or len(prefix) != 1:
        raise HTTPException(status_code=400, detail="prefix는 한 글자여야 합니다.")

    _ensure_ready()
    res = compute_health_single(prefix.lower(), _F_new, _calib)  # 없으면 None

    return {
        "prefix": prefix.lower(),
        "result": res,
        "meta": {
            "agg_percentile": AGG_P,
            "last_seconds": LAST_SECONDS,
            "use_log": USE_LOG,
        },
    }
