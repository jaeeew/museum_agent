// C:/Exhibit/curator_server/frontend/src/components/Welcome.jsx

import React, { useState } from "react"
import { useNavigate } from "react-router-dom"

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001"

export default function Welcome() {
  const navigate = useNavigate()
  const [intent, setIntent] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  // ✅ 새로 추가: 그냥 갤러리로 보내는 버튼
  const handleOpenGallery = () => {
    // 기본 회화 갤러리
    navigate("/gallery?category=painting_json")
    // 만약 카테고리 없이 전체 갤러리라면:
    // navigate("/gallery")
  }

  const handleStart = async () => {
    const q = intent.trim() || "오늘 볼 만한 작품을 추천해줘"
    if (loading) return

    setLoading(true)
    setError("")

    try {
      const res = await fetch(`${API}/ai/agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          // 필요하면 기본 카테고리 힌트도 보낼 수 있음
          // category: "painting_json",
        }),
      })

      if (!res.ok) {
        const msg = await res.text().catch(() => "")
        throw new Error(msg || `Agent 호출 실패 (HTTP ${res.status})`)
      }

      const agent = await res.json()
      console.log("agent result:", agent)

      const action = agent.action || "curate"
      const primaryId = agent.primary_id
      const secondaryId = agent.secondary_id
      const category = agent.category || "painting_json"

      // action별 라우팅
      if (action === "compare" && primaryId && secondaryId) {
        // 두 작품 비교 화면
        navigate(
          `/compare?ids=${encodeURIComponent(primaryId)},${encodeURIComponent(
            secondaryId
          )}&category=${encodeURIComponent(category)}`
        )
      } else if (action === "tts" && primaryId) {
        // 🧠 TTS 중심 몰입형 관람 → Immersive 페이지로 이동
        navigate(
          `/immersive?id=${encodeURIComponent(
            primaryId
          )}&category=${encodeURIComponent(category)}`
        )
      } else if (primaryId) {
        // 일반 상세(텍스트 중심) 보기
        navigate(
          `/detail/${encodeURIComponent(
            primaryId
          )}?category=${encodeURIComponent(category)}`
        )
      } else {
        // 아무 작품도 결정 못한 경우 → 갤러리로 fallback
        navigate(`/gallery?intent=${encodeURIComponent(q)}`)
      }
    } catch (e) {
      console.error(e)
      setError(
        "AI 큐레이터 연결 중 문제가 발생했습니다. 갤러리에서 직접 작품을 선택해 주세요."
      )
      // 에러가 나도 사용자가 막히지 않게 갤러리로 보냄
      navigate(`/gallery?intent=${encodeURIComponent(q)}`)
    } finally {
      setLoading(false)
    }
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
            disabled={loading}
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
              opacity: loading ? 0.6 : 1,
            }}
          />

          {error && (
            <div
              style={{
                marginTop: "8px",
                fontSize: "13px",
                color: "#b91c1c",
              }}
            >
              {error}
            </div>
          )}

          {/* 하단: 오른쪽 버튼 두 개 (갤러리 / AI 큐레이터) */}
<div
  style={{
    marginTop: "14px",
    display: "flex",
    alignItems: "center",
    justifyContent: "flex-end",
    gap: "8px",
  }}
>
  {/* 🔍 그냥 작품 목록만 보고 싶을 때 */}
  <button
    type="button"
    onClick={handleOpenGallery}
    disabled={loading}
    style={{
      padding: "9px 16px",
      borderRadius: "999px",
      border: "1px solid #d1d5db",
      backgroundColor: "white",
      color: "#374151",
      fontWeight: 500,
      fontSize: "14px",
      cursor: loading ? "default" : "pointer",
    }}
  >
    전체 작품 갤러리 보기
  </button>

  {/* 🤖 기존 AI 큐레이터 버튼 */}
  <button
    type="button"
    onClick={handleStart}
    disabled={loading}
    style={{
      padding: "9px 18px",
      borderRadius: "999px",
      border: "none",
      backgroundColor: loading ? "#e5e7eb" : "#e2b48a",
      color: "#332218",
      fontWeight: 600,
      fontSize: "14px",
      cursor: loading ? "default" : "pointer",
      whiteSpace: "nowrap",
      boxShadow: loading
        ? "none"
        : "0 8px 20px rgba(139,92,55,0.35)",
    }}
  >
    {loading ? "AI 큐레이터가 전시를 준비 중..." : "전시장으로 입장하기"}
  </button>
</div>

        </div>
      </div>
    </div>
  )
}
