# app/routers/runrisk.py
from fastapi import APIRouter, HTTPException, Query,  Path
from typing import Dict, Any
from app.db import get_cursor
from app.run_RUL_NEW import compute_rul_for_sample
from pydantic import BaseModel

router = APIRouter(prefix="/runrisk", tags=["runrisk"])

class RulSampleResponse(BaseModel):
    mode: int
    blade_id: int
    message: str = "값이 정상적으로 들어왔는지 확인용"

@router.get("/mse")
def get_mse_by_mode_blade(
        mode: int = Query(..., ge=1),
        blade_id: int = Query(..., ge=0),
        limit: int = Query(5000, ge=1, le=100000),
) -> Dict[str, Any]:
    """
    예) GET /runrisk/mse?mode_id=1&blade_id=Blade-001
    반환: { count, rows:[{time_stamp:<ISO>, mse:<float>}, ...] }
    """
    sql = """
          SELECT timestamp, mse
          FROM run_risk
          WHERE mode = %s
            AND blade_id = %s
          ORDER BY timestamp ASC
              LIMIT %s
          """
    try:
        with get_cursor() as cur:
            cur.execute(sql, (mode, blade_id, limit))
            print(f"[rul_sample_test] mode={mode}, blade_id={blade_id}", flush=True)
            rows = []
            for r in cur.fetchall():
                ts = r.get("timestamp")
                iso = ts.isoformat(sep=" ", timespec="seconds") if hasattr(ts, "isoformat") else str(ts)
                val = r.get("mse")
                rows.append({"timestamp": iso, "mse": float(val) if val is not None else None})
        return {"count": len(rows), "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

# # 2) ✅ 경로 방식: /runrisk/rul_sample/1/224
# @router.get("/rul_sample/{mode}/{blade_id}", response_model=RulSampleResponse)
# def rul_sample_path(
#         mode: int = Path(..., ge=1, description="1 이상"),
#         blade_id: int = Path(..., ge=0, description="0 이상"),
# ):
#     print(f"[rul_sample_path] mode={mode}, blade_id={blade_id}", flush=True)
#     return RulSampleResponse(mode=mode, blade_id=blade_id)

# 2) ✅ 경로 방식: /runrisk/rul_sample/1/224

@router.get("/rul_sample")
def rul_sample_path(
        mode: int = Query(..., ge=1),
        blade_id: int = Query(..., ge=0),
) -> Dict[str, Any]:
    try:
        print("📌 요청된 mode:", mode)
        print("📌 요청된 blade_id:", blade_id)
        result = compute_rul_for_sample(mode = mode, blade_id = blade_id)
        print("✅ 라우터에서 호출된 함수 compute_rul_for_sample")
        print(result)
        # result는 {"health_now": float|None, "rul_days": float|None, "pred_end_date": "YYYY-MM-DD"|None}
        return {
            "mode": mode,
            "blade_id": blade_id,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

