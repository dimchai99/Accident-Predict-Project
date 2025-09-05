import json, zipfile, io, os, sys
# 원본/결과 경로 수정해 쓰세요
SRC = r"C:\Users\1\Accident-Predict-Project\python\autoencoder_v1.keras"
DST = r"C:\Users\1\Accident-Predict-Project\python\autoencoder_v1_fixed.keras"

def wrap_triplet_if_flat(v):
    # ['x',0,0] 처럼 납작 리스트면 [['x',0,0]] 로 감싸기
    if isinstance(v, list) and v and isinstance(v[0], (str, int)):
        return [v]
    return v

def fix_inbound_nodes(inbound):
    # 기대형태: [[ ['prev',0,0,{}], ... ], ... ]
    # 케이스1) ['prev',0,0,{}] 납작 => [[ ['prev',0,0,{}] ]]
    if isinstance(inbound, list) and inbound and isinstance(inbound[0], (str, int)):
        return [[inbound]]
    # 케이스2) [ ['prev',0,0,{}], ... ] 1단계 부족 => [ [ ['prev',0,0,{}], ... ] ]
    if isinstance(inbound, list) and inbound and isinstance(inbound[0], list) and inbound and inbound != []:
        # 안쪽 원소가 triplet/quad 형태라면 한 단계 감싸기
        first = inbound[0]
        if first and isinstance(first[0], (str, int)):
            return [inbound]
    return inbound

with zipfile.ZipFile(SRC, "r") as z:
    files = {n: z.read(n) for n in z.namelist()}
    if "config.json" not in files:
        raise RuntimeError("config.json not found in .keras archive")

cfg = json.loads(files["config.json"].decode("utf-8"))

# 모델 설정 노드 찾기
model_cfg = cfg.get("model_config", cfg)
if "config" in model_cfg:
    mc = model_cfg["config"]
else:
    mc = model_cfg

# input/output_layers 포맷 보정
for key in ("input_layers", "output_layers"):
    if key in mc:
        mc[key] = wrap_triplet_if_flat(mc[key])

# 각 layer의 inbound_nodes 보정
layers = mc.get("layers", [])
for L in layers:
    conf = L.get("config", {})
    if "inbound_nodes" in L:
        L["inbound_nodes"] = fix_inbound_nodes(L["inbound_nodes"])
    # 혹시 layer config 하단에 있을 경우까지 방어
    if "inbound_nodes" in conf:
        conf["inbound_nodes"] = fix_inbound_nodes(conf["inbound_nodes"])

# 다시 write
files["config.json"] = json.dumps(cfg, ensure_ascii=False).encode("utf-8")
with zipfile.ZipFile(DST, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for name, data in files.items():
        z.writestr(name, data)

print("Wrote fixed .keras ->", DST)
