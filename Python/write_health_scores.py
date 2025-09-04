import json
from datetime import datetime, timezone
import numpy as np
import pymysql
import joblib
from tensorflow import keras
import os

# --- 1) DB 설정 읽기 ---
def load_db_config(path="db_config.txt"):
    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    cfg["port"] = int(cfg.get("port", "3306"))
    return cfg

CFG = load_db_config()

# --- 2) 모델/스케일러 로드 ---
SCALER_PATH = "standard_scaler_v1.joblib"
MODEL_PATH = "autoencoder_v1.keras"

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"{SCALER_PATH} not found")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found")

scaler = joblib.load(SCALER_PATH)
ae = keras.models.load_model(MODEL_PATH, compile=False)

# --- 3) 더미 입력 생성(실전에서는 cycle_feature.features SELECT해서 X 생성) ---
# shape: (N, D) → 스케일러와 모델이 학습된 입력 차원에 맞춰야 함
N, D = 128, scaler.n_features_in_
X_raw = np.random.randn(N, D)
X = scaler.transform(X_raw)
X_hat = ae.predict(X, verbose=0)
mse = np.mean((X - X_hat) ** 2, axis=1)
thr = float(np.mean(mse) + 3 * np.std(mse))
states = np.where(mse < thr, "healthy", "warning")

# --- 4) DB 연결 ---
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

# --- 5) pipeline_run 기록(선택) ---
cur.execute(
    """
    INSERT INTO pipeline_run (model_version, scaler_version, status, params, notes)
    VALUES (%s, %s, 'running', %s, %s)
    """,
    ("v1", "v1", json.dumps({"thr": thr}), "demo run"),
)
run_id = cur.lastrowid

# --- 6) health_score insert ---
sql = """
      INSERT INTO health_score
      (asset_id, component_id, blade_replacement_id, ts, score, rul_cycles, state, method, params, source, run_id)
      VALUES (%s,%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s)
      """

now = datetime.now(timezone.utc).replace(tzinfo=None)
rows = []
for s, st in zip(mse, states):
    rows.append(
        (
            1,              # asset_id (있으면 실제 값으로)
            None,           # component_id
            None,           # blade_replacement_id
            float(s),       # score (0=건강 ~ 1=위험 의미로 사용)
            None,           # rul_cycles (있으면 입력)
            st,             # state
            "autoencoder_mse",
            json.dumps({"thr": thr}),
            "python_batch",
            run_id,
        )
    )
cur.executemany(sql, rows)

# 파이프라인 상태 업데이트
cur.execute("UPDATE pipeline_run SET status='success', ended_at=NOW() WHERE run_id=%s", (run_id,))

cur.close()
conn.close()
print(f"Inserted {len(rows)} rows into health_score (run_id={run_id}).")

# DB 연결 후 cursor(cur) 만든 다음에 이 함수 추가/사용
def get_or_create_asset_id(cursor, line_name="demo_line"):
    cursor.execute("SELECT asset_id FROM asset WHERE line_name=%s LIMIT 1", (line_name,))
    row = cursor.fetchone()
    if row:
        return int(row[0])
    cursor.execute("INSERT INTO asset(line_name) VALUES (%s)", (line_name,))
    return int(cursor.lastrowid)

asset_id = get_or_create_asset_id(cur, "demo_line")

# rows 만들 때 1 대신 asset_id 사용
# ...
rows.append((
    asset_id,  # ← 여기!
    None, None,
    # ts는 이미 NOW()로 SQL에 넣고 있으면 생략
    float(s),
    None,
    st,
    "autoencoder_mse",
    json.dumps({"thr": thr}),
    "python_batch",
    run_id,
))

