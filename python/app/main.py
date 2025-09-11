# app/main.py
# FastAPI 서버 엔트리 포인트
# /health → 서버 살아있는지 확인
# /inference/series/{prefix} → 단일 prefix로 추론
# /inference/series_batch → 여러 prefix 한꺼번에 추론
# DB에서 불러온 데이터를 model_runner.infer_df()로 처리하고 결과를 benchmark_score 테이블에 저장
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from app.db import get_cursor
from app.model_runner import infer_df
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
# main.py
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, mse, blades, runrisk

from app.db import DB_CONFIG  # load_dotenv 포함되어 있음
import pymysql

print("===== FastAPI MAIN.PY LOADED =====")

app = FastAPI()
# CORS 설정: 프론트엔드와 백엔드가 다른 주소에서 통신할 수 있도록 허용
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:3003",# React 개발 서버 주소
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:3002",
    "http://127.0.0.1:8003",
    "http://127.0.0.1:3003",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(mse.router)
app.include_router(blades.router)
app.include_router(runrisk.router)

@app.get("/mse/by_prefix/{prefix}")
def get_mse_by_prefix(prefix: str, limit: Optional[int] = None):
    print("DEBUG: get_mse_by_prefix called with prefix =", prefix)
    """
    benchmark_mse에서 blade_inbound_id가 'prefix'로 시작하는 행만 반환
    예: /mse/by_prefix/a -> a1, a2, ... 만
    """
    try:
        sql = """
              SELECT blade_inbound_id, time_stamp, mse
              FROM benchmark_mse
              WHERE LEFT(blade_inbound_id, 1) = %s
              ORDER BY blade_inbound_id ASC, time_stamp ASC
              """
        params = [prefix]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))

        with get_cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            print("DEBUG rows:", rows[:5])

        return {
            "prefix": prefix,
            "count": len(rows),
            "rows": rows,
        }

    except Exception as e:
        # DB 연결 오류 포함 모든 예외 처리
        raise HTTPException(status_code=500, detail=f"Database error: {e}")