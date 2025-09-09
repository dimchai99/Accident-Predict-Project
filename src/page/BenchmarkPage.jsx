// src/page/BenchmarkPage.jsx
import React from "react";
import { Link } from "react-router-dom"; // 라우터 사용 시
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

// 더미 데이터 (라인차트용)
const lineData = [
    { year: 2015, furniture: 10000, technology: 18000, office: 5000 },
    { year: 2016, furniture: 20000, technology: 22000, office: 10000 },
    { year: 2017, furniture: 30000, technology: 28000, office: 15000 },
    { year: 2018, furniture: 32000, technology: 30000, office: 17000 },
    { year: 2019, furniture: 28000, technology: 34000, office: 20000 },
];

// 더미 데이터 (바차트용)
const barData = [
    { name: "Jan", bar: 4000 },
    { name: "Feb", bar: 5000 },
    { name: "Mar", bar: 6000 },
    { name: "Apr", bar: 8000 },
    { name: "May", bar: 10000 },
    { name: "Jun", bar: 15000 },
];

// 더미 데이터 (칼날 재고 확인용)
const tableData = [
    { id: "Blade-001", position: "System Architect", startDate: "2011/04/25" },
    { id: "Blade-002", position: "Accountant", startDate: "2011/07/25" },
    { id: "Blade-003", position: "Junior Technical Author", startDate: "2009/01/12" },
    { id: "Blade-004", position: "Senior Javascript Developer", startDate: "2012/03/29" },
    { id: "Blade-005", position: "Accountant", startDate: "2008/11/28" },
];

export default function Benchmark() {
    return (
        <div className="benchmark-page">
            {/* 상단 네이비 바 */}
            <div className="top-bar">
                {/* ✅ 홈 버튼 (메인화면 이동) */}
                <Link to="/" className="home-button">
                    <i className="fas fa-home"></i>
                </Link>
                <h1 className="title ms-3">Accident Prediction Monitoring System</h1>
            </div>

            <div className="layout">
                {/* ✅ 왼쪽 사이드바 → 검색창 + 테이블 */}
                <aside className="sidebar">
                    {/* 🔎 검색창 */}
                    <div className="sidebar-search mb-3">
                        <div className="input-group">
                            <input
                                type="text"
                                className="form-control"
                                placeholder="Search for..."
                                aria-label="Search"
                            />
                            <button className="btn btn-primary" type="button">
                                <i className="fas fa-search"></i>
                            </button>
                        </div>
                    </div>

                    {/* 📊 칼날 재고 확인 */}
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
                            {/* 🔹 첫 번째 행만 보여줌 */}
                            <tr>
                                <td>{tableData[0].id}</td>
                                <td>{tableData[0].position}</td>
                            </tr>
                            </tbody>
                        </table>
                    </div>
                </aside>

                {/* ✅ 오른쪽 콘텐츠 */}
                <main className="content container-fluid px-4">
                    {/* 페이지 제목 + 서브 박스 */}
                    <h1 className="page-title">Benchmark</h1>
                    <div className="page-subtitle">Dashboard</div>

                    <div className="row mt-4">
                        {/* ✅ 라인차트 카드 */}
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
                                            <Line type="monotone" dataKey="furniture" stroke="#1e3a8a" strokeWidth={2} />
                                            <Line type="monotone" dataKey="technology" stroke="#f59e0b" strokeWidth={2} />
                                            <Line type="monotone" dataKey="office" stroke="#ef4444" strokeWidth={2} />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        {/* ✅ 바차트 카드 */}
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
                                            <Line type="monotone" dataKey="bar" stroke="#1e3a8a" strokeWidth={2} />
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
