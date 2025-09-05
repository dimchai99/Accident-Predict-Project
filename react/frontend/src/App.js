import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Homepage from './page/homepage';
import BenchmarkPage from "./page/BenchmarkPage";


function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<BenchmarkPage />} />
            </Routes>
        </Router>
    );
}

export default App;
