import React from "react";
import { Link } from "react-router-dom";
import { motion, useScroll, useTransform } from "framer-motion";
import factoryImg from "../image/factory.png";
import icons2 from "../image/icon2.png"; // 톱니 이미지
import "../index.css";

export default function MainPage() {
    const { scrollY } = useScroll();
    const y = useTransform(scrollY, [0, 500], [0, 400]);
    const scale = useTransform(scrollY, [0, 500], [1, 1.4]);

    return (
        <div className="page-container">
            {/* 상단 네이비 바 */}
            <div className="top-bar">
                <div className="circle"></div>
                <div className="circle"></div>
                <div className="circle"></div>
            </div>

            {/* 패럴랙스 이미지 */}
            <div className="header-image">
                <motion.img
                    src={factoryImg}
                    alt="factory"
                    className="w-full h-full object-cover"
                    style={{ y, scale }}
                />
            </div>

            {/* 버튼 + 톱니바퀴 영역 */}
            <motion.div
                className="button-section"
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                viewport={{ once: true, amount: 0.3 }}
            >
                {/* 왼쪽 큰 톱니 */}
                <img src={icons2} alt="gear" className="gear gear-left" />

                {/* 버튼 */}
                <Link to="/benchmark" className="custom-button">
                    Benchmark
                </Link>
                <Link to="/active-monitor" className="custom-button">
                    Active Monitor
                </Link>

                {/* 오른쪽 중간 + 작은 톱니 */}
                <img src={icons2} alt="gear" className="gear gear-right" />
                <img src={icons2} alt="gear" className="gear gear-small" />
            </motion.div>
        </div>
    );
}
