import React from "react"
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom"
import Gallery from "./components/Gallery"
import Detail from "./components/Detail"
import Compare from "./components/Compare"
import "./index.css"   // ← 이 줄 꼭 있어야 함

export default function App() {
  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center text-center">
              <h1 className="text-4xl font-bold text-blue-600 mb-6">🎨 AI Curator</h1>
              <p className="text-gray-600 mb-8">문화유산과 예술 작품을 AI가 큐레이션합니다.</p>
              <Link to="/gallery" className="bg-blue-500 text-white px-6 py-3 rounded-2xl hover:bg-blue-600 transition">
                갤러리 보기
              </Link>
            </div>
          }
        />
        <Route path="/gallery" element={<Gallery />} />
        <Route path="/detail/:id" element={<Detail />} />
        <Route path="/compare" element={<Compare />} />   {/* ✅ 추가 */}
        <Route path="*" element={<div className="p-8">404 Not Found</div>} />
      </Routes>
    </Router>
  )
}

