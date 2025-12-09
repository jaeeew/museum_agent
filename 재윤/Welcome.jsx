// C:/Exhibit/curator_server/frontend/src/components/Welcome.jsx

import React, { useState } from "react"
import { useNavigate } from "react-router-dom"

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001"

export default function Welcome() {
  const navigate = useNavigate()
  const [intent, setIntent] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  // 🔍 전체 갤러리 버튼 (카테고리까지 주고 싶으면 아래처럼)
  const handleOpenGallery = () => {
    navigate("/gallery?category=painting_json")
    // 전체 통합 갤러리라면 ↓ 로 바꿔도 됨
    // navigate("/gallery")
  }

  // 🤖 AI 에이전트 호출 후 라우팅
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
          // 필요하면 기본 카테고리 힌트도 같이 보낼 수 있음
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

      if (action === "compare" && primaryId && secondaryId) {
        // 두 작품 비교 화면
        navigate(
          `/compare?ids=${encodeURIComponent(primaryId)},${encodeURIComponent(
            secondaryId
          )}&category=${encodeURIComponent(category)}`
        )
      } else if (action === "tts" && primaryId) {
        // TTS 몰입형 화면
        navigate(
          `/immersive?id=${encodeURIComponent(
            primaryId
          )}&category=${encodeURIComponent(category)}`
        )
      } else if (primaryId) {
        // 일반 상세 화면
        navigate(
          `/detail/${encodeURIComponent(
            primaryId
          )}?category=${encodeURIComponent(category)}`
        )
      } else {
        // 아무 작품도 못 고르면 → 에러만 띄우고 멈추기
        setError("이 요청에 맞는 작품을 찾지 못했습니다. 다른 표현으로 적어볼까요?")
      }
    } catch (e) {
      console.error(e)
      setError(
        "AI 큐레이터 연결 중 문제가 발생했습니다. 갤러리에서 직접 작품을 선택해 주세요."
      )
      
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
      {/* 배경 (두 번째 UX 버전) */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: "url('/museum-bg.jpg')",
          backgroundSize: "cover",
          backgroundPosition: "center center",
          filter: "blur(1.2px)",
          transform: "scale(1.03)",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "linear-gradient(to bottom, rgba(248,244,237,0.85), rgba(248,244,237,0.96))",
        }}
      />

      {/* 메인 컨텐츠 */}
      <div
        style={{
          position: "relative",
          zIndex: 1,
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "40px 16px",
          boxSizing: "border-box",
          flexDirection: "column",
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "28px", maxWidth: 900 }}>
          <h1
            style={{
              margin: 0,
              marginBottom: "10px",
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

          {/* 한줄 설명 */}
          <p
            style={{
              margin: 0,
              marginTop: "8px",
              fontSize: 14,
              color: "#4b3c32",
            }}
          >
            오늘 어떤 전시를 보고 싶은지 자유롭게 적어 주세요.
            <br />
            에이전트가 어울리는 작품 탐색 / 해설 / 비교 방식을 골라 드립니다.
          </p>

          {/* 기능 안내 문구 */}
          <p
            style={{
              margin: 0,
              marginTop: "12px",
              fontSize: 15,
              color: "#4b3c32",
              lineHeight: 1.5,
            }}
          >
            ✨ 이 전시장에서는 <strong>작품 설명</strong>, <strong>TTS 음성 해설</strong>,
            <strong> 작품 비교</strong> 기능을 사용할 수 있어요.
            <br />
            원하는 내용을 입력하면 AI가 가장 적합한 방식으로 안내합니다.
          </p>

          {/* 기능 안내 박스 */}
          <div
            style={{
              marginTop: "16px",
              display: "inline-block",
              textAlign: "left",
              padding: "12px 18px",
              borderRadius: "18px",
              backgroundColor: "rgba(255,255,255,0.7)",
              boxShadow: "0 10px 26px rgba(148,112,70,0.18)",
              fontSize: 13,
              color: "#5b4634",
              lineHeight: 1.7,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 4 }}>
              ✨ 이 전시장에서 할 수 있는 일
            </div>
            <div>🖋️ <strong>작품 설명 에이전트</strong> — 작품의 시대·배경·의미를 해설합니다.</div>
            <div>🎧 <strong>TTS 음성 해설</strong> — 작품 속으로 들어가듯 몰입형 해설을 제공합니다.</div>
            <div>🔍 <strong>작품 비교 에이전트</strong> — 두 작품의 스타일·색감·상징을 분석합니다.</div>
          </div>
        </div>

        {/* 입력 카드 */}
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
          <textarea
            value={intent}
            onChange={(e) => setIntent(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              "예시)\n- 잔잔한 분위기의 동양화 전시를 추천해줘\n- 두 작품을 비교해서 차이점을 알고 싶어\n- 오늘 기분에 어울리는 작품을 골라줘"
            }
            disabled={loading}
            style={{
              width: "100%",
              border: "none",
              outline: "none",
              resize: "none",
              fontSize: "15px",
              backgroundColor: "transparent",
              padding: "8px 10px",
              minHeight: "72px",
              borderRadius: "12px",
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

          <div
            style={{
              marginTop: "14px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <div
              style={{
                fontSize: 12,
                color: "#8b7350",
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span>💡</span>
              <span>엔터로 전시장 입장 · Shift+Enter 줄바꿈</span>
            </div>

            <div
              style={{
                display: "flex",
                gap: 10,
                flexWrap: "wrap",
              }}
            >
              <button
                type="button"
                onClick={handleOpenGallery}
                disabled={loading}
                style={{
                  padding: "8px 16px",
                  borderRadius: "999px",
                  backgroundColor: "rgba(255,255,255,0.9)",
                  color: "#6b4c33",
                  fontSize: "13px",
                  cursor: loading ? "default" : "pointer",
                  boxShadow: "0 4px 16px rgba(148,112,70,0.18)",
                  border: "none",
                }}
              >
                🖼 전체 작품 갤러리 보기
              </button>

              <button
                type="button"
                onClick={handleStart}
                disabled={loading}
                style={{
                  padding: "9px 22px",
                  borderRadius: "999px",
                  background: loading
                    ? "linear-gradient(135deg, #d1d5db, #9ca3af)"
                    : "linear-gradient(135deg, #c9925d, #b37540)",
                  color: "white",
                  fontSize: "14px",
                  cursor: loading ? "default" : "pointer",
                  whiteSpace: "nowrap",
                  boxShadow: loading
                    ? "none"
                    : "0 8px 20px rgba(139,92,55,0.35)",
                  border: "none",
                }}
              >
                {loading ? "AI 큐레이터가 전시를 준비 중..." : "🎟️ 전시장으로 입장하기"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
