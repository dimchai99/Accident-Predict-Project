# app/routers/blades.py
from fastapi import APIRouter, HTTPException
from app.db import get_cursor

router = APIRouter(prefix="/blades", tags=["blades"])

@router.get("/by_mode/{mode_id}")
def get_blades_by_mode(mode_id: int):
    """
    run_risk 테이블에서 mode_id = {mode_id} 인 레코드의
    blade_id를 중복 없이 반환
    """
    try:
        with get_cursor() as cur:
            cur.execute("""
                        SELECT DISTINCT blade_id
                        FROM run_risk
                        WHERE mode = %s
                        ORDER BY blade_id ASC
                        """, (mode_id,))
            rows = cur.fetchall()

        return {"mode_id": mode_id, "blade_ids": [r["blade_id"] for r in rows]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


