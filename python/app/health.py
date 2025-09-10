# app/routers/health.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
import json
import os

# 헬스 계산 유틸 재사용
from app.benchmark_health import (
    try_load_artifacts, build_reference_from_new, build_ecdf_smooth,
    compute_health_single, USE_LOG, AGG_P, LAST_SECONDS,
)

router = APIRouter(prefix="/health", tags=["health"])

# ── lazy 초기화: ECDF/캘리브레이션 한번만 로드 ──────────────────────────
_F_new = None
_calib = None

def _ensure_ready():
    global _F_new, _calib
    if _F_new is not None and _calib is not None:
        return
    loaded = try_load_artifacts()
    if loaded is not None:
        F_new, CALIB = loaded
        calib = dict(
            H_MAX=CALIB["H_MAX"], H_MIN=CALIB["H_MIN"],
            TAU=CALIB["TAU"], p0=CALIB["p0"], b=CALIB["b"], gamma=CALIB["gamma"],
        )
        _F_new, _calib = F_new, calib
    else:
        # 저장된 아티팩트가 없을 때 NEW 데이터로 ECDF를 즉석 구성
        scores_new_ref, _ = build_reference_from_new()
        _F_new = build_ecdf_smooth(scores_new_ref, alpha=0.5, beta=0.5)
        _calib = dict(H_MAX=95.0, H_MIN=5.0, TAU=0.007, p0=0.5, b=0.0, gamma=1.0)

# ── JSONL 디버그 로그 (원하면 끄거나 경로 변경) ─────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DEBUG_DB_PATH = Path(os.getenv("DEBUG_DB_PATH") or ROOT / "python" / "debug_db.jsonl")
DEBUG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _append_jsonl(obj: dict):
    with DEBUG_DB_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ── (A) 단일 prefix: 프런트에서 사용하는 API ──────────────────────────
@router.get("/by_prefix/{prefix}")
def get_health_by_prefix(prefix: str) -> Dict:
    """
    프런트 요구사항: /health/by_prefix/{prefix}
    응답 형태: { prefix, result: {health, agg, p, p_shift} | null, meta: {...} }
    """
    if not prefix or len(prefix) != 1:
        raise HTTPException(status_code=400, detail="prefix는 한 글자여야 합니다.")

    _ensure_ready()
    res = compute_health_single(prefix.lower(), _F_new, _calib)  # None | dict

    payload = {
        "prefix": prefix.lower(),
        "result": res,  # None이면 프런트에서 '—'로 표현하게 됨
        "meta": {
            "agg_percentile": AGG_P,
            "last_seconds": LAST_SECONDS,
            "use_log": USE_LOG,
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": "health.by_prefix",
        },
    }
    _append_jsonl(payload)
    return payload

# ── (B) 여러 prefix 요약(옵션) ─────────────────────────────────────────
@router.get("/summary")
def get_health_summary(prefixes: Optional[str] = Query(None, description="콤마로 구분 (예: a,b,c)")) -> Dict:
    _ensure_ready()
    targets: List[str] = [p.strip().lower() for p in prefixes.split(",")] if prefixes else []
    rows = []
    for px in targets:
        if not px or len(px) != 1:
            continue
        res = compute_health_single(px, _F_new, _calib)
        rows.append({"prefix": px, "result": res})

    payload = {
        "meta": {
            "agg_percentile": AGG_P,
            "last_seconds": LAST_SECONDS,
            "use_log": USE_LOG,
            "count": len(rows),
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": "health.summary",
        },
        "rows": rows,
    }
    _append_jsonl(payload)
    return payload
