# app/routers/mse.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from app.db import get_cursor   # ✅ db.py 재사용

router = APIRouter(prefix="/mse", tags=["mse"])

@router.get("/by_prefix/{prefix}")
def get_mse_by_prefix(prefix: str) -> Dict[str, Any]:
    """
    GET /mse/by_prefix/{prefix}
    응답: { prefix, rows: [{time_stamp, mse}], count }
    - blade_inbound_id 앞자리가 prefix인 레코드만 조회
    - time_stamp 오름차순
    """
    if not prefix or len(prefix) != 1:
        raise HTTPException(status_code=400, detail="prefix는 한 글자여야 합니다.")

    like_prefix = f"{prefix.lower()}%"

    # ⚠️ benchmark_mse.time_stamp 컬럼은 스키마상 DOUBLE 이므로 isoformat 변환 불필요
    sql = """
          SELECT time_stamp, mse
          FROM benchmark_mse
          WHERE LOWER(blade_inbound_id) LIKE %s
          ORDER BY time_stamp ASC \
          """

    with get_cursor() as cur:
        cur.execute(sql, (like_prefix,))
        rows: List[Dict[str, Any]] = cur.fetchall()

    return {
        "prefix": prefix.lower(),
        "rows": rows,            # [{time_stamp: <float>, mse: <float>}, ...]
        "count": len(rows),
    }
