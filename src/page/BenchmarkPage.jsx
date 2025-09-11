// src/page/BenchmarkPage.jsx
import React, { useState } from "react";
import { Link } from "react-router-dom";
import "../Benchmark.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "@fortawesome/fontawesome-free/css/all.min.css";
import sortIcon from "../image/sort.png";

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

// 백엔드 주소 (.env에서 REACT_APP_PY_BASE로 덮어쓰기 가능)
const BASE_URL = process.env.REACT_APP_PY_BASE || "http://127.0.0.1:8000";

// Blade-001 -> 'a', Blade-002 -> 'b' ... (6개 주기 반복)
const bladeToPrefix = (bladeId) => {
    const m = String(bladeId).match(/\d+/);
    const num = m ? parseInt(m[0], 10) : 0; // Blade-001 -> 1
    const letters = ["a", "b", "c", "d", "e", "f"];
    return num > 0 ? letters[(num - 1) % letters.length] : null;
};

// health → 메시지
const healthMsg = (h) => {
    if (h == null) return "데이터 없음";
    if (h >= 70) return "칼날 상태 양호";
    return "불량";
};

// 테이블 데이터 (Id만 필요)
const tableData = [
    { id: "Blade-001" },
    { id: "Blade-002" },
    { id: "Blade-003" },
    { id: "Blade-004" },
    { id: "Blade-005" },

];

export default function Benchmark() {
    const [searchTerm, setSearchTerm] = useState("");
    const [filteredData, setFilteredData] = useState(tableData);

    // 점수/상태
    const [selectedId, setSelectedId] = useState("");
    const [lastPrefix, setLastPrefix] = useState(null);
    const [health, setHealth] = useState(null);
    const [loading, setLoading] = useState(false);
    const [err, setErr] = useState("");

    // MSE 라인차트
    const [mseRows, setMseRows] = useState([]);
    const [mseLoading, setMseLoading] = useState(false);
    const hasMse = mseRows && mseRows.length > 0;

    // 백엔드 호출: /health/by_prefix/{prefix}
    const fetchHealth = async (prefix) => {
        setLoading(true);
        setErr("");
        setLastPrefix(prefix);
        try {
            const res = await fetch(`${BASE_URL}/health/by_prefix/${prefix}`);
            if (!res.ok) {
                const txt = await res.text();
                throw new Error(`HTTP ${res.status} ${txt}`);
            }
            const data = await res.json();
            setHealth(data?.result?.health ?? null);
        } catch (e) {
            setErr(e.message || "fetch failed");
            setHealth(null);
        } finally {
            setLoading(false);
        }
    };

    // 백엔드 호출: /mse/by_prefix/{prefix}
    const fetchMse = async (prefix) => {
        setMseLoading(true);
        setErr("");
        try {
            const res = await fetch(`${BASE_URL}/mse/by_prefix/${prefix}`);
            if (!res.ok) {
                const txt = await res.text();
                throw new Error(`HTTP ${res.status} ${txt}`);
            }
            const data = await res.json();
            const rows = (data?.rows || []).map((d) => {
                const iso = String(d.time_stamp || "");
                const label = iso.replace("T", " ").slice(5, 16);
                return {
                    time: label,
                    mse: Number(d.mse),
                    _ts_raw: iso,
                };
            });
            setMseRows(rows);
        } catch (e) {
            setErr(e.message || "fetch failed");
            setMseRows([]);
        } finally {
            setMseLoading(false);
        }
    };

// 검색 핸들러: 검색 시 결과만 갱신 (페이지 리셋 같은 건 필요 없음)
    const handleSearch = () => {
        const term = searchTerm.trim().toLowerCase();
        if (!term) {
            setFilteredData(tableData);
            return;
        }
        const result = tableData.filter((row) =>
            row.id.toLowerCase().includes(term)   // 부분 일치
        );
        setFilteredData(result);
    };

    const handleRowClick = (id) => {
        setSearchTerm(id);
        setSelectedId(id);
        const prefix = bladeToPrefix(id);
        if (prefix) {
            fetchHealth(prefix);
            fetchMse(prefix);
        } else {
            setHealth(null);
            setLastPrefix(null);
            setMseRows([]);
        }
    };

    const healthClass = (h) => {
        if (h == null) return "text-muted";
        if (h >= 70) return "text-primary";   // 매우 양호 → 초록
        return "text-danger";                 // 교체 권고 → 빨강
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
                                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                            />
                            <button className="btn btn-primary" onClick={handleSearch}>
                                <i className="fas fa-search"></i>
                            </button>
                        </div>
                    </div>

                    <div className="table-responsive">
                        <div className="list-group">
                            {filteredData.length > 0 ? (
                                filteredData.map((row, index) => (
                                    <button
                                        key={index}
                                        onClick={() => handleRowClick(row.id)}
                                        className={`list-group-item list-group-item-action ${
                                            selectedId === row.id ? "active" : ""
                                        }`}
                                        title="클릭해서 Health 지수 보기"
                                    >
                                        {row.id}
                                    </button>
                                ))
                            ) : (
                                <div className="list-group-item text-center text-muted">No Data</div>
                            )}
                        </div>
                    </div>

                </aside>

                {/* 콘텐츠 */}
                <main className="content">
                    <h1 className="page-title">Benchmark</h1>
                    {/* 수정된 부제목 */}
                    <div className="page-subtitle">
                        Dashboard
                    </div>

                    {/* 점수 + 메시지 */}
                    <div className="d-flex gap-3 mb-4">
                        <div className="card equal-card">
                            <div className="card-header custom-card-header">
                                <i className="fas fa-star me-1"></i> Health 점수
                            </div>
                            <div className="card-body text-center">
                                {loading ? (
                                    <p className="text-secondary m-0">불러오는 중…</p>
                                ) : err ? (
                                    <p className="text-danger m-0">{err}</p>
                                ) : (
                                    <>
                                        <p className="display-6 text-primary m-0">
                                            {health == null ? "—" : `${health.toFixed(2)}%`}
                                        </p>
                                        <small className="text-muted">
                                            {selectedId
                                                ? `ID: ${selectedId}`
                                                : "테이블에서 ID를 클릭하세요"}
                                        </small>
                                    </>
                                )}
                            </div>
                        </div>
                        <div className="card equal-card">
                            <div className="card-header custom-card-header">
                                <i className="fas fa-comment me-1"></i> 메시지
                            </div>
                            <div className="card-body text-center" style={{display:'flex', justifyContent:'centrt', alignItems:'center'}}>
                                {loading ? (
                                    <p className="text-secondary m-0">불러오는 중…</p>
                                ) : err ? (
                                    <h2 className="text-danger m-0">{err}</h2>
                                ) : (
                                    <h2 className={`lead m-0 ${healthClass(health)}`}>
                                        {healthMsg(health)}
                                    </h2>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* 그래프 */}
                    <div className="card mb-4 chart-card">
                        <div className="card-header custom-card-header">
                            <i className="fas fa-chart-line me-1"></i>
                            MSE 라인차트 (time_stamp vs mse)
                        </div>
                        <div className="card-body">
                            {mseLoading ? (
                                <div className="text-secondary">그래프 불러오는 중…</div>
                            ) : hasMse ? (
                                <ResponsiveContainer width="100%" height={300}>
                                    <ComposedChart data={mseRows}>
                                        <CartesianGrid stroke="#f5f5f5" />
                                        <XAxis dataKey="time" minTickGap={28} />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Line
                                            type="monotone"
                                            dataKey="mse"
                                            name="MSE"
                                            stroke="#1e3a8a"
                                            strokeWidth={3}
                                            dot={false}
                                        />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="text-muted">
                                    데이터가 없습니다. 왼쪽에서 Blade ID를 클릭해 주세요.
                                </div>
                            )}
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}
