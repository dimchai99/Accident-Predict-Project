// src/page/BenchmarkPage.jsx
import React, { useState } from "react";
import { Link } from "react-router-dom";
import "../Benchmark.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "@fortawesome/fontawesome-free/css/all.min.css";

import {
    ComposedChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";

// 🔑 백엔드 주소
const BASE_URL = process.env.REACT_APP_PY_BASE || "http://127.0.0.1:8000";

// 차트 데이터 (샘플)
const lineData = [
    { year: 2015, furniture: 10000, technology: 18000, office: 5000 },
    { year: 2016, furniture: 20000, technology: 22000, office: 10000 },
    { year: 2017, furniture: 30000, technology: 28000, office: 15000 },
    { year: 2018, furniture: 32000, technology: 30000, office: 17000 },
    { year: 2019, furniture: 28000, technology: 34000, office: 20000 },
];

// 테이블 데이터 (mode 번호 포함)
const tableData = [
    { id: "mode-001", mode: 1 },
    { id: "mode-002", mode: 2 },
    { id: "mode-003", mode: 3 },
    { id: "mode-004", mode: 4 },
    { id: "mode-005", mode: 5 },
    { id: "mode-006", mode: 6 },
    { id: "mode-007", mode: 7 },
    { id: "mode-008", mode: 8 },
];

export default function ActiveMonitor() {

    const [selectedMode, setSelectedMode] = useState(null);
    const [selectedBlade, setSelectedBlade] = useState(null);
    const [runRows, setRunRows] = useState([]);     // [{time, mse, _ts_raw}]
    const [runLoading, setRunLoading] = useState(false);
    const [runErr, setRunErr] = useState("");
    const [searchTerm, setSearchTerm] = useState("");
    const [filteredData, setFilteredData] = useState(tableData);
    const [isDashOpen, setIsDashOpen] = useState(true);

    // 🔑 새로 추가된 상태값
    const [openMode, setOpenMode] = useState(null); // 현재 열려 있는 모드
    const [bladeIds, setBladeIds] = useState({}); // mode별 blade_id 목록 캐시
    const [loading, setLoading] = useState(false);
    const [err, setErr] = useState("");

    const [healthNow, setHealthNow] = useState(null);
    const [rulDays, setRulDays] = useState(null);
    const [predEnd, setPredEnd] = useState(null);

    const fetchRulSample = async (mode, bladeId) => {
        const TAG = "[fetchRulSample]";
        console.group(`${TAG} start`);
        try {
            // 0) 입력값/환경 체크
            console.log(TAG, "args:", { mode, bladeId, BASE_URL: typeof BASE_URL !== "undefined" ? BASE_URL : "(undefined)" });

            const encMode = encodeURIComponent(mode);
            const encBlade = encodeURIComponent(bladeId);
            const url = `${BASE_URL}/runrisk/rul_sample?mode=${encMode}&blade_id=${encBlade}`;
            console.log(TAG, "built url:", url);

            // 1) fetch 호출
            console.time(`${TAG} fetch`);
            const res = await fetch(url, { method: "GET", cache: "no-store" });
            console.timeEnd(`${TAG} fetch`);

            console.log(TAG, "response.ok:", res.ok, "status:", res.status, res.statusText);
            console.log(TAG, "response.url:", res.url, "type:", res.type, "redirected:", res.redirected);

            // 1-1) 헤더 덤프
            const headersDump = {};
            try { res.headers.forEach((v, k) => (headersDump[k] = v)); } catch {}
            console.log(TAG, "response headers:", headersDump);
            const contentType = res.headers.get("content-type");
            console.log(TAG, "content-type:", contentType);

            // 2) 에러 응답 처리
            if (!res.ok) {
                const txt = await res.text().catch(e => `<<failed to read body: ${e}>>`);
                console.log(TAG, "non-OK body:", txt);
                throw new Error(`HTTP ${res.status} ${res.statusText} :: ${txt}`);
            }

            // 3) JSON 파싱
            console.time(`${TAG} parse json`);
            const data = await res.json();
            console.timeEnd(`${TAG} parse json`);
            console.log(TAG, "raw data:", data, "typeof data:", typeof data);

            // 4) 타입/숫자 변환
            const hn = Number(data.health_now);
            const rd = Number(data.rul_days);
            const pe = data.pred_end_date ?? null;

            console.log(TAG, "parsed numbers:", {
                health_now_raw: data.health_now,
                rul_days_raw: data.rul_days,
                health_now_num: hn,
                rul_days_num: rd,
                isFinite_hn: Number.isFinite(hn),
                isFinite_rd: Number.isFinite(rd),
                pred_end_date_raw: data.pred_end_date
            });

            // 5) 상태 업데이트 (리액트 setState는 비동기 → '스케줄됨'으로 로깅)
            const nextHN = Number.isFinite(hn) ? hn : null;
            const nextRD = Number.isFinite(rd) ? rd : null;

            console.log(TAG, "setHealthNow() scheduling:", nextHN);
            setHealthNow(nextHN);

            console.log(TAG, "setRulDays() scheduling:", nextRD);
            setRulDays(nextRD);

            console.log(TAG, "setPredEnd() scheduling:", pe);
            setPredEnd(pe);

            // 6) 최종 스냅샷
            console.log(TAG, "final scheduled values:", {
                health_now: nextHN,
                rul_days: nextRD,
                pred_end_date: pe
            });

        } catch (e) {
            console.error(`${TAG} ERROR:`, e?.stack || e);
            // 실패 시 상태 리셋
            try {
                console.log(TAG, "resetting states to null due to error");
                setHealthNow(null);
                setRulDays(null);
                setPredEnd(null);
            } catch (stateErr) {
                console.error(TAG, "state reset failed:", stateErr);
            }
        } finally {
            console.groupEnd(`${TAG} start`);
        }
    };


    // 검색
    const handleSearch = () => {
        if (searchTerm.trim() === "") {
            setFilteredData(tableData);
            console.log("filterdata : ", tableData);
        } else {
            const result = tableData.filter(
                (row) => row.id.toLowerCase() === searchTerm.toLowerCase()
            );
            setFilteredData(result.length > 0 ? result : []);
        }
    };

    // 모드 클릭 → Blade 목록 가져오기
    const toggleMode = async (row) => {
        if (openMode === row.id) {
            // 이미 열려 있으면 닫기
            setOpenMode(null);
            return;
        }

        setOpenMode(row.id); // 열림 상태는 id 기준
        setErr("");
        setLoading(true);

        try {
            const res = await fetch(`${BASE_URL}/blades/by_mode/${row.mode}`);
            if (!res.ok) {
                const txt = await res.text();
                throw new Error(`HTTP ${res.status} ${txt}`);
            }
            const data = await res.json();
            setBladeIds((prev) => ({ ...prev, [row.id]: data.blade_ids || [] }));
            console.log("bladeId: ",data);
        } catch (e) {
            setErr(e.message || "fetch failed");
            setBladeIds((prev) => ({ ...prev, [row.id]: [] }));
        } finally {
            setLoading(false);
        }
    };

    const fetchRunRiskMse = async (modeId, bladeId) => {
        setRunLoading(true);
        setRunErr("");
        try {
            const url = `${BASE_URL}/runrisk/mse?mode=${encodeURIComponent(modeId)}&blade_id=${encodeURIComponent(bladeId)}`;
            const res = await fetch(url);
            if (!res.ok) {
                const txt = await res.text();
                throw new Error(`HTTP ${res.status} ${txt}`);
            }
            const data = await res.json();
            console.log("[/runrisk/mse] raw:", data);

            // 1) 숫자만 남기기 (NaN 제거)  2) 라벨 만들기
            const rows = (data?.rows || [])
                .map((d) => {
                    const iso = String(d.time_stamp || "");
                    const label = iso.replace("T", " ").slice(5, 16); // "MM-DD HH:MM"
                    const mseNum = Number(d.mse);
                    return { time: label, mse: mseNum, _ts_raw: iso };
                })
                .filter((r) => Number.isFinite(r.mse)); // ← NaN/undefined 제거

            console.log("[/runrisk/mse] parsed:", rows.length, rows.slice(0, 5));
            setRunRows(rows);
        } catch (e) {
            setRunErr(e.message || "fetch failed");
            setRunRows([]);
        } finally {
            setRunLoading(false);
        }
    };
    return (
        <div className="benchmark-page">
            {/* ✅ Bootstrap 네이비 Navbar */}
            <nav
                className="navbar navbar-dark"
                style={{ backgroundColor: "#0f1b46", height: "50px" }}
            >
                <div className="container-fluid">
                    <Link to="/" className="navbar-brand d-flex align-items-center">
                        <i className="fas fa-home me-2"></i>
                        Accident Prediction Monitoring
                    </Link>
                </div>
            </nav>

            <div className="layout">
                {/* 사이드바 */}
                <aside className="sidebar">
                    <div className="sidebar-search mb-3">
                        <div className="input-group">
                            <input
                                type="text"
                                className="form-control"
                                placeholder="Search for..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                            />
                            <button className="btn btn-primary" onClick={handleSearch}>
                                <i className="fas fa-search"></i>
                            </button>
                        </div>
                    </div>

                    <div className="list-group">
                        {filteredData.length > 0 ? (
                            filteredData.map((row, index) => (
                                <div key={index} className="mb-2">
                                    <button
                                        onClick={() => toggleMode(row)}
                                        className={`list-group-item list-group-item-action ${
                                            openMode === row.id ? "active" : ""
                                        }`}
                                    >
                                        <i
                                            className={`me-2 fas fa-chevron-${
                                                openMode === row.id ? "down" : "right"
                                            }`}
                                        />
                                        {row.id}
                                    </button>


                                    {/* 하위 Blade ID 목록 */}
                                    {openMode === row.id && (
                                        <div className="ms-4 mt-2">
                                            {loading ? (
                                                <div className="text-secondary">불러오는 중...</div>
                                            ) : err ? (
                                                <div className="text-danger">{err}</div>
                                            ) : bladeIds[row.id]?.length > 0 ? (
                                                <ul className="list-group">
                                                    {bladeIds[row.id].map((b) => (
                                                        <button
                                                            key={b}
                                                            className="list-group-item list-group-item-action"
                                                            onClick={() => {
                                                                setSelectedMode(row.mode);
                                                                setSelectedBlade(b);
                                                                fetchRunRiskMse(row.mode, b);
                                                                fetchRulSample(row.mode, b);
                                                            }}
                                                            title="이 블레이드의 MSE 시계열 보기"
                                                        >
                                                            <i className="fas fa-microchip me-2 text-primary" />
                                                            {b}
                                                        </button>
                                                    ))}
                                                </ul>
                                            ) : (
                                                <div className="text-muted">데이터 없음</div>
                                            )}
                                        </div>
                                    )}

                                </div>
                            ))
                        ) : (
                            <div className="list-group-item text-center text-muted">
                                No Data
                            </div>
                        )}
                    </div>
                </aside>

                {/* 콘텐츠 */}
                <main className="content">
                    <h1 className="page-title">ActiveMonitor</h1>
                    <div
                        className="page-subtitle d-flex align-items-center user-select-none"
                        role="button"
                        aria-expanded={isDashOpen}
                        aria-controls="dashboard-section"
                        tabIndex={0}
                        onClick={() => setIsDashOpen((o) => !o)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                setIsDashOpen((o) => !o);
                            }
                        }}
                    >
                        <i
                            className={`me-2 fas fa-chevron-${isDashOpen ? "down" : "right"}`}
                            aria-hidden="true"
                        />
                        <span>ActiveMonitor</span>
                    </div>
                    {isDashOpen && (

                        <section id="dashboard-section">
                            {/* 점수 + 메시지 */}
                            <div className="d-flex gap-3 mb-4">
                                <div className="card equal-card">
                                    <div className="card-header custom-card-header">
                                        <i className="fas fa-star me-1"></i> 점수
                                    </div>
                                    <div className="card-body text-center">
                                        <h3 className="text-primary">
                                           {healthNow !== null ? `${healthNow.toFixed(1)}` : "-"}
                                        </h3>
                                    </div>
                                </div>
                                <div className="card equal-card">
                                    <div className="card-header custom-card-header">
                                        <i className="fas fa-comment me-1"></i> RUL일수
                                    </div>
                                    <div className="card-body text-center">
                                        <h3 className="text-primary">
                                        {typeof rulDays === "number" ? `${rulDays.toFixed(1)}일` : "-"}
                                        </h3>
                                    </div>
                                </div>
                                <div className="card equal-card">
                                    <div className="card-header custom-card-header">
                                        <i className="fas fa-comment me-1"></i> 예상 교체일
                                    </div>
                                    <div className="card-body text-center">
                                        <h3 className="text-primary">
                                        {predEnd ?? "-"}
                                        </h3>
                                    </div>
                                </div>
                            </div>
                            {/* RUN RISK: MSE 테이블 & 라인차트 */}
                            <div className="card mb-4">
                                <div className="card-header custom-card-header d-flex justify-content-between align-items-center">
                                    <span>
                                      <i className="fas fa-table me-1"></i>
                                      Run MSE (mode_id: {selectedMode ?? "-"}, blade_id: {selectedBlade ?? "-"})
                                    </span>
                                    {runRows.length > 0 && (
                                        <small className="text-muted">{runRows.length} rows</small>
                                    )}
                                </div>

                                <div className="card-body">
                                    {/* 표 */}
                                    {runLoading ? (
                                        <div className="text-secondary">불러오는 중…</div>
                                    ) : runErr ? (
                                        <div className="text-danger">{runErr}</div>
                                    ) : runRows.length === 0 ? (
                                        <div className="text-muted">좌측에서 모드 → 블레이드를 선택하세요.</div>
                                    ) : (
                                        <>


                                            {/* 라인차트 */}
                                            <ResponsiveContainer
                                                key={`${selectedMode}-${selectedBlade}-${runRows.length}`} // ← 컨테이너 0px 문제 대비
                                                width="100%"
                                                height={280}
                                            >
                                                <ComposedChart data={runRows}>
                                                    <CartesianGrid stroke="#f5f5f5" />
                                                    <XAxis dataKey="time" minTickGap={28} />
                                                    <YAxis />
                                                    <Tooltip
                                                        formatter={(value) => (Number.isFinite(value) ? value.toFixed(6) : value)}
                                                        labelFormatter={(label) => `Time: ${label}`}
                                                    />
                                                    <Legend />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="mse"
                                                        name="MSE"
                                                        stroke="#1e3a8a"
                                                        strokeWidth={3}
                                                        dot={false}
                                                        isAnimationActive={false} // 초기 렌더 안정화 (원하면 true로)
                                                    />
                                                </ComposedChart>
                                            </ResponsiveContainer>

                                        </>
                                    )}
                                </div>
                            </div>
                        </section>
                    )}
                </main>
            </div>
        </div>
    );
}
