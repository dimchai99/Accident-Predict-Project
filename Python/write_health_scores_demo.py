# write_health_scrores_demo.py
import json
import numpy as np
import pymysql

# --- DB 설정(로컬) ---
CFG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "1234",
    "database": "accident_db",
}

LINE_NAME = "demo_line"   # 이 라인 네임의 asset_id를 찾아 사용

# --- 1) 더미 입력 생성 ---
N, D = 10, 5    # 샘플 10개, 피처 5개 가정
X = np.random.randn(N, D)
mse = np.mean(X**2, axis=1)
thr = float(np.mean(mse) + 3 * np.std(mse))
states = np.where(mse < thr, "healthy", "warning")

# --- 2) DB 연결 ---
conn = pymysql.connect(
    host=CFG["host"],
    port=CFG["port"],
    user=CFG["user"],
    password=CFG["password"],
    database=CFG["database"],
    charset="utf8mb4",
    autocommit=True,
)
cur = conn.cursor()

# --- 유틸: asset_id 조회/생성 ---
def get_or_create_asset_id(cursor, line_name: str) -> int:
    cursor.execute("SELECT asset_id FROM asset WHERE line_name=%s LIMIT 1", (line_name,))
    row = cursor.fetchone()
    if row:
        return int(row[0])
    cursor.execute("INSERT INTO asset(line_name) VALUES (%s)", (line_name,))
    return int(cursor.lastrowid)

asset_id = get_or_create_asset_id(cur, LINE_NAME)

# --- 3) pipeline_run 기록 ---
cur.execute(
    """
    INSERT INTO pipeline_run (model_version, scaler_version, status, params, notes)
    VALUES (%s, %s, 'running', %s, %s)
    """,
    ("demo", "demo", json.dumps({"thr": thr}), "demo run"),
)
run_id = cur.lastrowid

# --- 4) health_score insert (ts는 DB NOW()) ---
sql = """
      INSERT INTO health_score
      (asset_id, component_id, blade_replacement_id, ts, score, rul_cycles, state, method, params, source, run_id)
      VALUES
          (%s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s) \
      """

rows = [
    (
        asset_id,       # 동적으로 조회한 asset_id 사용
        None,           # component_id
        None,           # blade_replacement_id
        float(s),       # score (원시 MSE; 운영에선 정규화 권장)
        None,           # rul_cycles
        st,             # state
        "dummy_mse",    # method
        json.dumps({"thr": thr}),
        "python_demo",
        run_id,
    )
    for s, st in zip(mse, states)
]

cur.executemany(sql, rows)

# --- 5) 파이프라인 상태 업데이트 ---
cur.execute(
    "UPDATE pipeline_run SET status='success', ended_at=NOW() WHERE run_id=%s",
    (run_id,),
)

cur.close()
conn.close()
print(f"[demo] asset_id={asset_id}, run_id={run_id}, inserted_rows={len(rows)}")
