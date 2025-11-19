// C:/Exhibit/curator_server/frontend/src/components/Welcome.jsx

import React, { useState } from "react"
import { useNavigate } from "react-router-dom"

export default function Welcome() {
  const navigate = useNavigate()
  const [intent, setIntent] = useState("")

  const handleStart = () => {
    const q = intent.trim() || "오늘 볼 만한 작품을 추천해줘"
    navigate(`/gallery?intent=${encodeURIComponent(q)}`)
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleStart()
    }
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        position: "relative",
        overflow: "hidden",
        fontFamily:
          "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color: "#2f2832",
      }}
    >
      {/* 🖼 박물관/궁 배경 + 크림 톤 오버레이 */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `
            linear-gradient(
              to bottom,
              rgba(255,255,255,0.55),
              rgba(250,249,246,0.75),
              rgba(245,242,236,0.9)
            ),
            url("/museum-bg.jpg")
          `,
          backgroundSize: "cover",
          backgroundPosition: "center center",
          backgroundRepeat: "no-repeat",
          filter: "saturate(1.05)",
          zIndex: 0,
        }}
      />

      {/* 중앙 레이아웃 */}
      <div
        style={{
          position: "relative",
          zIndex: 1,
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "40px 16px",
          boxSizing: "border-box",
        }}
      >
        {/* 타이틀 한 줄 */}
        <div style={{ textAlign: "center", marginBottom: "28px" }}>
          <h1
            style={{
              margin: 0,
              marginBottom: "6px",
              fontSize: "clamp(30px, 4vw, 40px)",
              fontWeight: 400,
              letterSpacing: "0.04em",
              color: "#2b2118",
              fontFamily:
                "'Nanum Myeongjo', 'Apple SD Gothic Neo', 'Malgun Gothic', serif",
            }}
          >
            AI 큐레이터 전시장에 오신 것을 환영합니다
          </h1>
        </div>

        {/* 카드형 입력 영역 */}
        <div
          style={{
            width: "100%",
            maxWidth: "1050px",
            borderRadius: "24px",
            backgroundColor: "rgba(253,251,247,0.96)",
            boxShadow: "0 18px 45px rgba(15, 23, 42, 0.22)",
            border: "1px solid rgba(0,0,0,0.04)",
            padding: "18px 22px",
            boxSizing: "border-box",
          }}
        >
          {/* 상단 입력창 */}
          <textarea
            value={intent}
            onChange={(e) => setIntent(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="오늘 어떤 전시를 보고 싶으신가요?"
            style={{
              width: "100%",
              border: "none",
              outline: "none",
              resize: "none",
              backgroundColor: "transparent",
              fontSize: "16px",
              lineHeight: 1.5,
              minHeight: "40px",
              maxHeight: "140px",
              color: "#312525",
            }}
          />

          {/* 하단: 오른쪽 버튼만 */}
          <div
            style={{
              marginTop: "14px",
              display: "flex",
              alignItems: "center",
              justifyContent: "flex-end", // ⬅ 오른쪽 정렬
            }}
          >
            <button
              type="button"
              onClick={handleStart}
              style={{
                padding: "9px 18px",
                borderRadius: "999px",
                border: "none",
                backgroundColor: "#e2b48a", // 연한 갈색
                color: "#332218",
                fontWeight: 600,
                fontSize: "14px",
                cursor: "pointer",
                whiteSpace: "nowrap",
                boxShadow: "0 8px 20px rgba(139,92,55,0.35)",
              }}
            >
              전시장으로 입장하기
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
