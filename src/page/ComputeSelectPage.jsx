// src/page/ComputeSelectPage.jsx
import React, { useState, useMemo } from "react";
import axios from "axios"; // 아직 안 쓸 거지만, 나중에 전송 시 사용
// 부트스트랩 쓰는 프로젝트라면 필요 시 주석 해제
// import "bootstrap/dist/css/bootstrap.min.css";

const BASE_URL = process.env.REACT_APP_PY_BASE || "http://127.0.0.1:8000";

// 예시: 모드 목록 / 샘플 목록
const ALL_MODES = [1, 2, 3, 4, 5, 6, 7, 8];
const ALL_SAMPLES = Array.from({ length: 10 }, (_, i) => `Blade-${String(i + 1).padStart(3, "0")}`);

export default function ComputeSelectPage() {
    const [modes, setModes] = useState([]);       // 선택된 mode 배열 (숫자)
    const [samples, setSamples] = useState([]);   // 선택된 sample 배열 (문자열)
    const [filter, setFilter] = useState("");     // sample 검색 필터
    const [payloadPreview, setPayloadPreview] = useState(null);
    const [err, setErr] = useState("");

    const filteredSamples = useMemo(() => {
        const q = filter.trim().toLowerCase();
        if (!q) return ALL_SAMPLES;
        return ALL_SAMPLES.filter(s => s.toLowerCase().includes(q));
    }, [filter]);

    const toggle = (arr, value, setArr) => {
        if (arr.includes(value)) setArr(arr.filter(v => v !== value));
        else setArr([...arr, value]);
    };

    const selectAllModes = () => setModes([...ALL_MODES]);
    const clearAllModes = () => setModes([]);

    const selectAllFilteredSamples = () => {
        // 화면에 보이는(필터 적용된) 샘플만 모두 선택
        const merged = new Set([...samples, ...filteredSamples]);
        setSamples([...merged]);
    };
    const clearAllSamples = () => setSamples([]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErr("");

        const payload = { modes, samples };
        setPayloadPreview(payload);
        console.log("[compute payload]", payload);

        // 나중에 백엔드 전송할 때:
        // try {
        //   const { data } = await axios.post(`${BASE_URL}/run_risk/compute`, payload);
        //   console.log("server response:", data);
        //   // TODO: 응답 처리 (화면 표시 등)
        // } catch (e) {
        //   setErr(e?.response?.data?.detail || e.message || "요청 실패");
        // }
    };

    return (
        <div className="container py-4">
            <h3 className="mb-3">모드 & 샘플 선택</h3>

            {/* MODES */}
            <section className="mb-4">
                <div className="d-flex align-items-center mb-2">
                    <h5 className="me-3 mb-0">Modes</h5>
                    <div className="btn-group btn-group-sm" role="group">
                        <button type="button" className="btn btn-outline-secondary" onClick={selectAllModes}>
                            전체선택
                        </button>
                        <button type="button" className="btn btn-outline-secondary" onClick={clearAllModes}>
                            해제
                        </button>
                    </div>
                </div>

                <div className="d-flex flex-wrap gap-3">
                    {ALL_MODES.map((m) => (
                        <label key={m} className="form-check-label" style={{ minWidth: 80 }}>
                            <input
                                type="checkbox"
                                className="form-check-input me-1"
                                checked={modes.includes(m)}
                                onChange={() => toggle(modes, m, setModes)}
                            />
                            mode {m}
                        </label>
                    ))}
                </div>
            </section>

            {/* SAMPLES */}
            <section className="mb-4">
                <div className="d-flex align-items-center mb-2">
                    <h5 className="me-3 mb-0">Samples</h5>
                    <input
                        type="text"
                        className="form-control form-control-sm me-2"
                        style={{ maxWidth: 220 }}
                        placeholder="검색 (예: 003)"
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                    />
                    <div className="btn-group btn-group-sm" role="group">
                        <button type="button" className="btn btn-outline-secondary" onClick={selectAllFilteredSamples}>
                            화면표시 전체선택
                        </button>
                        <button type="button" className="btn btn-outline-secondary" onClick={clearAllSamples}>
                            해제
                        </button>
                    </div>
                </div>

                <div className="d-flex flex-wrap gap-3">
                    {filteredSamples.map((s) => (
                        <label key={s} className="form-check-label" style={{ minWidth: 120 }}>
                            <input
                                type="checkbox"
                                className="form-check-input me-1"
                                checked={samples.includes(s)}
                                onChange={() => toggle(samples, s, setSamples)}
                            />
                            {s}
                        </label>
                    ))}
                </div>
            </section>

            {/* SUBMIT */}
            <button
                className="btn btn-primary"
                onClick={handleSubmit}
                disabled={!modes.length || !samples.length}
            >
                선택값 담기 (미리보기)
            </button>

            {err && <div className="text-danger mt-3">{err}</div>}

            {/* PREVIEW */}
            {payloadPreview && (
                <div className="mt-4">
                    <h6>Payload Preview</h6>
                    <pre className="p-3 bg-light border rounded" style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(payloadPreview, null, 2)}
          </pre>
                </div>
            )}
        </div>
    );
}