# app/db.py
# -----------------------------------------------------------------------------
# 📌 DB 연결 관리 모듈
#
# 기능
#  - .env 파일에서 MySQL 접속 정보를 읽어 DB_CONFIG에 저장
#  - contextmanager(get_cursor)로 커넥션/커서를 안전하게 열고 닫음
#  - DictCursor를 사용해 SELECT 결과를 dict 형태로 반환
#
# 역할
#  - FastAPI 서비스 전반에서 DB 연결을 표준화
#  - 쿼리 실행 시 with 구문으로 간단하게 사용:
#
#       with get_cursor() as cur:
#           cur.execute("SELECT * FROM blade_benchmark LIMIT 5")
#           rows = cur.fetchall()
#
#  - 코드 어디서 실행하더라도 루트/.env를 강제로 로드하므로 환경변수 안정적
# -----------------------------------------------------------------------------

import os, pymysql
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path

# ✅ 루트/.env를 강제로 지정 (어디서 실행해도 안정적으로 로드)
ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트
load_dotenv(ROOT / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST","127.0.0.1"),
    "port": int(os.getenv("DB_PORT","3306")),
    "user": os.getenv("DB_USER","root"),
    "password": os.getenv("DB_PASSWORD","1234"),
    "database": os.getenv("DB_NAME","accident_db"),
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": True,
    "charset": "utf8mb4",
}

@contextmanager
def get_cursor():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            yield cur
    finally:
        conn.close()
