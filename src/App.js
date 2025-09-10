// src/App.js
import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainPage from './page/MainPage';
import Benchmark from "./page/BenchmarkPage";
import ActiveMonitor from "./page/ActiveMonitor";

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<MainPage />} />
                <Route path="/benchmark" element={<Benchmark />} />
                <Route path="/active-monitor" element={<ActiveMonitor />} />
            </Routes>
        </Router>
    );
}

export default App;
