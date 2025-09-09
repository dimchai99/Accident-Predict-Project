// src/page/BenchmarkPage.jsx
import React from "react";
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

export default function Benchmark() {
    return (
        <div className="benchmark-page">
            {/* 상단 네이비 바 */}
            <div className="top-bar">
                <h1 className="title">팀이름</h1>
            </div>

            <div className="layout">
                {/* 왼쪽 사이드바 */}
                <aside className="sidebar">
                    <ul>
                        <li>대시보드</li>
                        <li>팀 관리</li>
                        <li>설정</li>
                    </ul>
                </aside>

                {/* 오른쪽 콘텐츠 */}
                <main className="content container-fluid px-4">
                    <div className="row">
                        {/* ✅ 라인차트 카드 (col-6) */}
                        <div className="col-xl-6">
                            <div className="card mb-4">
                                <div className="card-header">
                                    <i className="fas fa-chart-line me-1"></i>
                                    벤치마킹 라인차트
                                </div>
                                <div className="card-body" style={{ height: "300px" }}>
                                    <ResponsiveContainer width="100%" height="100%">
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
                                                strokeWidth={2}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="technology"
                                                stroke="#f59e0b"
                                                strokeWidth={2}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="office"
                                                stroke="#ef4444"
                                                strokeWidth={2}
                                            />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        {/* ✅ 바차트 카드 (col-6) */}
                        <div className="col-xl-6">
                            <div className="card mb-4">
                                <div className="card-header">
                                    <i className="fas fa-chart-bar me-1"></i>
                                    바 차트
                                </div>
                                <div className="card-body" style={{ height: "300px" }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={barData}>
                                            <CartesianGrid stroke="#f5f5f5" />
                                            <XAxis dataKey="name" />
                                            <YAxis />
                                            <Tooltip />
                                            <Legend />
                                            <Bar dataKey="bar" barSize={40} fill="#82ca9d" />
                                            <Line
                                                type="monotone"
                                                dataKey="bar"
                                                stroke="#1e3a8a"
                                                strokeDasharray="5 5"
                                                strokeWidth={2}
                                            />
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
