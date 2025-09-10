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
    if (h >= 85) return "칼날 상태 매우 양호";
    if (h >= 60) return "양호";
    if (h >= 35) return "점검 필요";
    return "교체 권고";
};

// 테이블 데이터 (Id만 필요)
const tableData = [
    { id: "Blade-001" },
    { id: "Blade-002" },
    { id: "Blade-003" },
    { id: "Blade-004" },
    { id: "Blade-005" },
    { id: "Blade-006" },
    { id: "Blade-007" },
    { id: "Blade-008" },
    { id: "Blade-009" },
    { id: "Blade-010" },
];

export default function Benchmark() {
    const [searchTerm, setSearchTerm] = useState("");
    const [filteredData, setFilteredData] = useState(tableData);
    const [currentPage, setCurrentPage] = useState(1);
    const rowsPerPage = 5;

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
            const data = await res.json(); // { prefix, result: { health, agg, p, p_shift } | null }
            setHealth(data?.result?.health ?? null);
        } catch (e) {
            setErr(e.message || "fetch failed");
            setHealth(null);
        } finally {
            setLoading(false);
        }
    };
    // ---- 백엔드 호출: /mse/by_prefix/{prefix}
    const fetchMse = async (prefix) => {
        setMseLoading(true);
        setErr("");
        try {
            const res = await fetch(`${BASE_URL}/mse/by_prefix/${prefix}`);
            if (!res.ok) {
                const txt = await res.text();
                throw new Error(`HTTP ${res.status} ${txt}`);
            }
            const data = await res.json(); // { rows: [{time_stamp, mse}], ... }
            const rows = (data?.rows || []).map((d) => {
                const iso = String(d.time_stamp || "");
                // X축 가독성: "MM-DD HH:mm" 포맷 느낌으로 잘라 보여주기
                const label = iso.replace("T", " ").slice(5, 16);
                return {
                    time: label, // XAxis dataKey
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

    const handleSearch = () => {
        if (searchTerm.trim() === "") {
            setFilteredData(tableData);
        } else {
            const result = tableData.filter(
                (row) => row.id.toLowerCase() === searchTerm.toLowerCase()
            );
            setFilteredData(result.length > 0 ? result : []);
            setCurrentPage(1);
        }
    };

    // 테이블 클릭 → prefix 계산 → 백엔드 호출 → 점수 갱신
    const handleRowClick = (id) => {
        setSearchTerm(id);
        setSelectedId(id);
        const prefix = bladeToPrefix(id);
        if (prefix) {
            fetchHealth(prefix);
            fetchMse(prefix);
        }
        else {
            setHealth(null);
            setLastPrefix(null);
            setMseRows([]);
        }
    };


    const indexOfLastRow = currentPage * rowsPerPage;
    const indexOfFirstRow = indexOfLastRow - rowsPerPage;
    const currentRows = filteredData.slice(indexOfFirstRow, indexOfLastRow);
    const totalPages = Math.ceil(filteredData.length / rowsPerPage);

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

                    <h5 className="mb-3">칼날 재고 확인</h5>
                    <div className="table-responsive">
                        <table className="table table-sm table-bordered">
                            <thead className="table-light">
                            <tr>
                                <th>Id</th>
                            </tr>
                            </thead>
                            <tbody>
                            {currentRows.length > 0 ? (
                                currentRows.map((row, index) => (
                                    <tr
                                        key={index}
                                        onClick ={()=> handleRowClick(row.id)}
                                        style={{ cursor: "pointer" }}
                                        title="클릭해서 Health 지수 보기"
                                    >
                                        <td>{row.id}</td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td className="text-center">No Data</td>
                                </tr>
                            )}
                            </tbody>
                        </table>
                    </div>

                    {totalPages > 1 && (
                        <nav>
                            <ul className="pagination justify-content-center">
                                {Array.from({ length: totalPages }, (_, i) => (
                                    <li
                                        key={i}
                                        className={`page-item ${
                                            currentPage === i + 1 ? "active" : ""
                                        }`}
                                    >
                                        <button
                                            className="page-link"
                                            onClick={() => setCurrentPage(i + 1)}
                                        >
                                            {i + 1}
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        </nav>
                    )}
                </aside>

                {/* 콘텐츠 */}
                <main className="content">
                    <h1 className="page-title">Benchmark</h1>
                    <div className="page-subtitle">Dashboard</div>

                    {/* 점수 + 메시지 (정사각형 카드) */}
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
                                                ? `ID: ${selectedId}${
                                                    lastPrefix ? ` → prefix: ${lastPrefix}%` : ""
                                                }`
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
                            <div className="card-body text-center">
                                <p className="lead">칼날 상태 양호</p>
                            </div>
                        </div>
                    </div>

                    {/* 그래프 (디자인 유지: ComposedChart + Grid + Legend) */}
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
