// src/App.jsx
import React from "react"
import { BrowserRouter as Router, Routes, Route } from "react-router-dom"

import Welcome from "./components/Welcome"   // 🏛 박물관 입구 느낌 첫 화면
import Gallery from "./components/Gallery"
import Detail from "./components/Detail"
import Compare from "./components/Compare"
import Immersive from "./components/Immersive"
import ArtworkGrid from "./components/ArtworkGrid" 

import "./index.css"   // ← Tailwind / 전역 스타일

export default function App() {
  return (
    <Router>
      <Routes>
        {/* 1. 박물관 입구 느낌의 첫 화면 */}
        <Route path="/" element={<Welcome />} />

        {/* 전체 작품 갤러리 (Welcome에서 '전체 작품 갤러리 보기' 누르면 여기로) */}
        <Route path="/gallery" element={<ArtworkGrid />} />

        {/* 2. 실제 작품 그리드 갤러리 */}
        <Route path="/gallery" element={<Gallery />} />

        {/* 3. 상세 / 비교 화면 */}
        <Route path="/detail/:id" element={<Detail />} />
        <Route path="/compare" element={<Compare />} />

        <Route path="/immersive" element={<Immersive />} />

        {/* 4. 404 */}
        <Route path="*" element={<div className="p-8">404 Not Found</div>} />
      </Routes>
    </Router>
  )
}
