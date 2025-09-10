# debug_db.py  — DB 연결/스키마/조회 점검(콘솔 출력)
import os, pymysql, json, sys, traceback
from pathlib import Path
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# app.db 의 설정 로직 그대로 재사용
ROOT = Path(__file__).resolve().parents[0]           # 프로젝트 루트(여기가 파일이 있는 위치)
PYDIR = ROOT / "python"                               # python 디렉토리
sys.path.insert(0, str(PYDIR))                        # app 모듈 import 가능하게
from app.db import DB_CONFIG  # load_dotenv 포함되어 있음

def mask(s, keep=2):
    if s is None: return None
    s = str(s)
    if len(s) <= keep: return '*'*len(s)
    return s[:keep] + '*'*(len(s)-keep)

print("=== DB CONFIG ===")
safe_cfg = {k:(mask(v) if k in ("password",) else v) for k,v in DB_CONFIG.items() if k!="cursorclass"}
print(json.dumps(safe_cfg, ensure_ascii=False, indent=2))

try:
    print("\n[1] Connecting...")
    conn = pymysql.connect(**DB_CONFIG)
    print(" -> Connected.")
    with conn.cursor() as cur:
        # 1) 현재 DB
        cur.execute("SELECT DATABASE()"); dbname = cur.fetchone()
        print(f"[2] DATABASE() = {dbname}")

        # 2) 테이블 존재 확인
        cur.execute("SHOW TABLES LIKE 'blade_benchmark'")
        tbl = cur.fetchone()
        print(f"[3] blade_benchmark exists? -> {bool(tbl)}")

        # 3) 행 수/미리보기
        if tbl:
            cur.execute("SELECT COUNT(*) AS cnt FROM blade_benchmark")
            print(f"[4] blade_benchmark COUNT = {cur.fetchone()['cnt']}")

            # prefix 한 번 체크 (a%)
            cur.execute("""
                        SELECT blade_benchmark_id
                              ,relative_timestamp
                              ,cut_torque
                              ,cut_lag_error
                              ,cut_speed
                              ,film_speed
                              ,film_lag_error
                        FROM blade_benchmark
                        WHERE blade_benchmark_id LIKE %s
                        ORDER BY relative_timestamp ASC, blade_benchmark_id ASC
                        """, ("e%",))
            rows = cur.fetchall()
            print(f"[5] sample rows for prefix 'a' ({len(rows)} rows):")
            # for r in rows:
            #     print(r)


            # ✅ DataFrame으로 변환
            df = pd.DataFrame(rows)

            # blade_benchmark_id 제외한 feature만 추출
            feature_cols = [c for c in df.columns if c not in ("blade_benchmark_id", "relative_timestamp")]

            # ✅ 스케일러 로드
            scaler = joblib.load("models/scaler.pkl")
            model  = tf.keras.models.load_model("models/fcast.h5",compile=False)


            # ✅ 스케일링
            X_scaled = scaler.transform(df[feature_cols].values)
            print('스케일 완료')
            ts_all   = df["relative_timestamp"].to_numpy()         # (N,)
            ids_all  = df["blade_benchmark_id"].to_numpy()         # (N,)

            L = 128    # 윈도우 길이
            STEP = 1   # stride
            N, p = X_scaled.shape
            if N <= L:
                raise RuntimeError(f"Not enough rows for L={L} (N={N})")


            pairs = []   # (blade_inbound_id, time_stamp, mse)
            last = N - L
            for s in range(0, last, STEP):
                x_win  = X_scaled[s:s+L]          # (L, p)
                y_true = X_scaled[s+L:s+L+1]      # (1, p)  다음 1스텝
                if y_true.size == 0:
                    break
                y_pred = model.predict(x_win[None, ...], verbose=0)  # (1, p)
                mse = float(np.mean((y_pred[0] - y_true[0])**2))
                # blade_benchmark_id 중 현재 시점의 ID + 타임스탬프 기록
                pairs.append((ids_all[s+L], float(ts_all[s+L]), mse))

            # ✅ 콘솔 출력
            print("\n[7] MSE results (앞부분 10개):")
            for rec in pairs[:5]:
                print({"blade_inbound_id": rec[0], "time_stamp": rec[1], "mse": rec[2]})

            if pairs:
                cur.executemany("""
                                INSERT INTO benchmark_mse (blade_inbound_id, time_stamp, mse)
                                VALUES (%s, %s, %s)
                                    ON DUPLICATE KEY UPDATE
                                                         mse = VALUES(mse)
                                """, [(str(bid), float(ts), float(mse)) for (bid, ts, mse) in pairs])

                print(f"[8] inserted/updated rows: {len(pairs)}")

                # (선택) 몇 개 확인용 조회
                cur.execute("""
                            SELECT blade_inbound_id, time_stamp, mse
                            FROM benchmark_mse
                            WHERE blade_inbound_id LIKE %s
                            ORDER BY time_stamp ASC
                                LIMIT 5
                            """, ("a%",))
                print("[9] sample from benchmark_mse:")
                for r in cur.fetchall():
                    print(r)
            else:
                print("[8] no pairs to insert")








        # # 4) benchmark_mse/benchmark_mse(혹은 benchmark_score) 구조 확인
        # for tname in ("benchmark_mse", "benchmark_score"):
        #     cur.execute(f"SHOW TABLES LIKE %s", (tname,))
        #     if cur.fetchone():
        #         cur.execute(f"SHOW CREATE TABLE {tname}")
        #         print(f"\n[6] SHOW CREATE TABLE {tname}:")
        #         print(cur.fetchone()["Create Table"])

    conn.close()
    print("\n✅ DONE.")
except Exception as e:
    print("\n❌ ERROR OCCURRED:")
    print(type(e).__name__, e)
    traceback.print_exc()
