// src/App.js
import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainPage from './page/MainPage';
import Benchmark from "./page/BenchmarkPage";   // ✅ Benchmark 컴포넌트 import

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<MainPage />} />
                <Route path="/benchmark" element={<Benchmark />} />
                <Route path="/active-monitor" element={<div>Active Monitor Page</div>} />
            </Routes>
        </Router>
    );
}

export default App;
