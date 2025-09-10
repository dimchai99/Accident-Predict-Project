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

// 차트 데이터
const lineData = [
    { year: 2015, furniture: 10000, technology: 18000, office: 5000 },
    { year: 2016, furniture: 20000, technology: 22000, office: 10000 },
    { year: 2017, furniture: 30000, technology: 28000, office: 15000 },
    { year: 2018, furniture: 32000, technology: 30000, office: 17000 },
    { year: 2019, furniture: 28000, technology: 34000, office: 20000 },
];

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

export default function ActiveMonitor() {
    const [searchTerm, setSearchTerm] = useState("");
    const [filteredData, setFilteredData] = useState(tableData);
    const [currentPage, setCurrentPage] = useState(1);
    const rowsPerPage = 5;

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
                                    <tr key={index}>
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
                    <h1 className="page-title">ActiveMonitor</h1>
                    <div className="page-subtitle">Dashboard</div>

                    {/* 점수 + 메시지 (정사각형 카드) */}
                    <div className="d-flex gap-3 mb-4">
                        <div className="card equal-card">
                            <div className="card-header custom-card-header">
                                <i className="fas fa-star me-1"></i> 점수
                            </div>
                            <div className="card-body text-center">
                                <p className="display-6 text-primary">85</p>
                            </div>
                        </div>
                        <div className="card equal-card">
                            <div className="card-header custom-card-header">
                                <i className="fas fa-comment me-1"></i> RUL
                            </div>
                            <div className="card-body text-center">
                                <p className="lead">칼날 상태 양호</p>
                            </div>
                        </div>
                    </div>

                    {/* 그래프 */}
                    <div className="card mb-4 chart-card">
                        <div className="card-header custom-card-header">
                            <i className="fas fa-chart-line me-1"></i>
                            벤치마킹 라인차트
                        </div>
                        <div className="card-body">
                            <ResponsiveContainer width="100%" height={300}>
                                <ComposedChart data={lineData}>
                                    <CartesianGrid stroke="#f5f5f5" />
                                    <XAxis dataKey="year" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="furniture"
                                        stroke="#1e3a8a"
                                        strokeWidth={3}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="technology"
                                        stroke="#f59e0b"
                                        strokeWidth={3}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="office"
                                        stroke="#ef4444"
                                        strokeWidth={3}
                                    />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}
