import React, { useState } from "react";
import { Link } from "react-router-dom";
import "../Benchmark.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "@fortawesome/fontawesome-free/css/all.min.css";

import {
    ComposedChart,
    Bar,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";

// 차트용 더미 데이터
const lineData = [
    { year: 2015, furniture: 10000, technology: 18000, office: 5000 },
    { year: 2016, furniture: 20000, technology: 22000, office: 10000 },
    { year: 2017, furniture: 30000, technology: 28000, office: 15000 },
    { year: 2018, furniture: 32000, technology: 30000, office: 17000 },
    { year: 2019, furniture: 28000, technology: 34000, office: 20000 },
];

const barData = [
    { name: "Jan", bar: 4000 },
    { name: "Feb", bar: 5000 },
    { name: "Mar", bar: 6000 },
    { name: "Apr", bar: 8000 },
    { name: "May", bar: 10000 },
    { name: "Jun", bar: 15000 },
];

// 칼날 재고 더미 데이터
const tableData = [
    { id: "Blade-001", position: "System Architect" },
    { id: "Blade-002", position: "Accountant" },
    { id: "Blade-003", position: "Junior Technical Author" },
    { id: "Blade-004", position: "Senior Javascript Developer" },
    { id: "Blade-005", position: "Accountant" },
    { id: "Blade-006", position: "Software Engineer" },
    { id: "Blade-007", position: "Integration Specialist" },
    { id: "Blade-008", position: "Sales Assistant" },
    { id: "Blade-009", position: "Manager" },
    { id: "Blade-010", position: "Data Scientist" },
];

export default function Benchmark() {
    const [searchTerm, setSearchTerm] = useState("");
    const [filteredData, setFilteredData] = useState(tableData);
    const [currentPage, setCurrentPage] = useState(1);
    const rowsPerPage = 5; // 🔹 한 페이지에 5개씩

    // 검색 버튼 클릭 시
    const handleSearch = () => {
        if (searchTerm.trim() === "") {
            setFilteredData(tableData); // 검색어 없으면 전체
        } else {
            const result = tableData.filter(
                (row) => row.id.toLowerCase() === searchTerm.toLowerCase()
            );
            setFilteredData(result.length > 0 ? result : []); // 없으면 공백
            setCurrentPage(1);
        }
    };

    // 페이지네이션 계산
    const indexOfLastRow = currentPage * rowsPerPage;
    const indexOfFirstRow = indexOfLastRow - rowsPerPage;
    const currentRows = filteredData.slice(indexOfFirstRow, indexOfLastRow);
    const totalPages = Math.ceil(filteredData.length / rowsPerPage);

    return (
        <div className="benchmark-page">
            {/* 상단 네이비 바 */}
            <div className="top-bar">
                <Link to="/" className="home-button">
                    <i className="fas fa-home"></i>
                </Link>
                <h1 className="title ms-3">Accident Prediction Monitoring System</h1>
            </div>

            <div className="layout">
                {/* ✅ 왼쪽 사이드바 */}
                <aside className="sidebar">
                    {/* 검색창 */}
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

                    {/* 칼날 재고 확인 */}
                    <h5 className="mb-3">칼날 재고 확인</h5>
                    <div className="table-responsive">
                        <table className="table table-sm table-bordered">
                            <thead className="table-light">
                            <tr>
                                <th>Id</th>
                                <th>Position</th>
                            </tr>
                            </thead>
                            <tbody>
                            {currentRows.length > 0 ? (
                                currentRows.map((row, index) => (
                                    <tr key={index}>
                                        <td>{row.id}</td>
                                        <td>{row.position}</td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="2" className="text-center">
                                        No Data
                                    </td>
                                </tr>
                            )}
                            </tbody>
                        </table>
                    </div>

                    {/* 페이지네이션 */}
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

                {/* 오른쪽 콘텐츠 */}
                <main className="content container-fluid px-4">
                    <h1 className="page-title">Benchmark</h1>
                    <div className="page-subtitle">Dashboard</div>

                    <div className="row mt-4">
                        {/* 라인차트 */}
                        <div className="col-xl-6">
                            <div className="card mb-4">
                                <div className="card-header">
                                    <i className="fas fa-chart-line me-1"></i>
                                    벤치마킹 라인차트
                                </div>
                                <div className="card-body">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={lineData}>
                                            <CartesianGrid stroke="#f5f5f5" />
                                            <XAxis dataKey="year" />
                                            <YAxis />
                                            <Tooltip />
                                            <Legend />
                                            <Line type="monotone" dataKey="furniture" stroke="#1e3a8a" />
                                            <Line type="monotone" dataKey="technology" stroke="#f59e0b" />
                                            <Line type="monotone" dataKey="office" stroke="#ef4444" />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        {/* 바차트 */}
                        <div className="col-xl-6">
                            <div className="card mb-4">
                                <div className="card-header">
                                    <i className="fas fa-chart-bar me-1"></i>
                                    바 차트
                                </div>
                                <div className="card-body">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={barData}>
                                            <CartesianGrid stroke="#f5f5f5" />
                                            <XAxis dataKey="name" />
                                            <YAxis />
                                            <Tooltip />
                                            <Legend />
                                            <Bar dataKey="bar" barSize={40} fill="#82ca9d" />
                                            <Line type="monotone" dataKey="bar" stroke="#1e3a8a" />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}
