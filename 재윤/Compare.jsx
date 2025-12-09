// C:/Exhibit/curator_server/frontend/src/components/Compare.jsx

import React, { useEffect, useMemo, useState } from "react"
import { useLocation, useNavigate } from "react-router-dom"

// Vite 사용 시 .env에서 VITE_API_BASE 지정 가능, 없으면 로컬 8000
const API = import.meta.env?.VITE_API_BASE || "http://127.0.0.1:8001"

export default function Compare() {
  const location = useLocation()
  const navigate = useNavigate()
  const params = useMemo(
    () => new URLSearchParams(location.search),
    [location.search]
  )

  const idsParam = params.get("ids") || ""
  const category = params.get("category") || "painting_json"
  const ids = idsParam
    .split(",")
    .map((s) => decodeURIComponent(s.trim()))
    .filter(Boolean)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [data, setData] = useState(null) // { left, right, analysis }

  // 🔹 같은 카테고리 내 모든 작품 id 목록 (랜덤 조합용)
  const [allIds, setAllIds] = useState([])
  const [loadingAlt, setLoadingAlt] = useState(false)

  // ---------------- 두 작품 비교 분석 요청 ----------------
  useEffect(() => {
    const controller = new AbortController()

    const run = async () => {
      if (ids.length !== 2) {
        setError("비교할 작품 ID 2개가 필요합니다.")
        setLoading(false)
        return
      }

      try {
        setLoading(true)
        setError("")

        const res = await fetch(`${API}/ai/analyze-compare`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids, category, locale: "ko" }),
          signal: controller.signal,
        })

        if (!res.ok) {
          const t = await res.text().catch(() => "")
          throw new Error(t || `분석 요청 실패 (HTTP ${res.status})`)
        }

        const json = await res.json()
        setData(json)
      } catch (e) {
        if (e.name !== "AbortError") {
          setError(e.message || "두 작품을 비교하는 중 문제가 발생했습니다.")
        }
      } finally {
        setLoading(false)
      }
    }

    run()
    return () => controller.abort()
  }, [idsParam, category])

  // ---------------- 카테고리별 전체 작품 ID 목록 로드 ----------------
  useEffect(() => {
    let cancelled = false

    const loadIds = async () => {
      try {
        const res = await fetch(`${API}/json_list/${category}`)
        if (!res.ok) return
        const files = await res.json() // ["kart_....json", ...]

        if (cancelled) return

        const idsOnly = files
          .map((name) => name.replace(/\.json$/i, ""))
          .filter(Boolean)

        setAllIds(idsOnly)
      } catch (e) {
        console.warn("json_list load failed:", e)
      }
    }

    loadIds()
    return () => {
      cancelled = true
    }
  }, [category])

  const retry = () => {
    navigate(0) // 쿼리 유지한 채 새로 분석
  }

  // 🔹 다른 조합으로 비교하기
  const handleOtherPair = () => {
    if (!allIds.length) return

    setLoadingAlt(true)

    // 지금 보고 있는 두 작품은 제외하고 랜덤 뽑기 시도
    const currentSet = new Set(ids)
    const pool = allIds.filter((id) => !currentSet.has(id))

    const source = pool.length >= 2 ? pool : allIds
    const shuffled = [...source].sort(() => Math.random() - 0.5)
    const nextPair = shuffled.slice(0, 2)

    const search = new URLSearchParams()
    search.set(
      "ids",
      nextPair.map((id) => encodeURIComponent(id)).join(",")
    )
    search.set("category", category)

    navigate(`/compare?${search.toString()}`)
  }

  // ---------------- 로딩/에러 화면 ----------------
  if (loading) {
    return (
      <div
        style={{
          minHeight: "100vh",
          background:
            "linear-gradient(to bottom, #fdfaf5 0%, #f5eee3 40%, #f5f3ee 100%)",
          padding: "32px 16px 40px",
          boxSizing: "border-box",
          fontFamily:
            "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        }}
      >
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <Header onBack={() => navigate(-1)} />
          <div
            style={{
              marginTop: 80,
              textAlign: "center",
              color: "#6b7280",
              fontSize: 16,
            }}
          >
            🔎 AI 큐레이터가 두 작품을 비교 분석하고 있습니다...
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div
        style={{
          minHeight: "100vh",
          background:
            "linear-gradient(to bottom, #fdfaf5 0%, #f5eee3 40%, #f5f3ee 100%)",
          padding: "32px 16px 40px",
          boxSizing: "border-box",
          fontFamily:
            "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        }}
      >
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <Header onBack={() => navigate(-1)} />

          <div
            style={{
              marginTop: 24,
              borderRadius: 20,
              border: "1px solid rgba(248,113,113,0.4)",
              backgroundColor: "#fef2f2",
              padding: "16px 18px",
              color: "#b91c1c",
              fontSize: 14,
              whiteSpace: "pre-wrap",
            }}
          >
            {error}
          </div>

          <div
            style={{
              marginTop: 16,
              display: "flex",
              gap: 10,
              flexWrap: "wrap",
            }}
          >
            <button
              onClick={retry}
              style={{
                padding: "8px 16px",
                borderRadius: 999,
                border: "none",
                backgroundColor: "#f3f4f6",
                fontSize: 14,
                cursor: "pointer",
              }}
            >
              다시 시도
            </button>
            <a
              href={`${API}/health`}
              target="_blank"
              rel="noreferrer"
              style={{
                padding: "8px 16px",
                borderRadius: 999,
                border: "1px solid #e5e7eb",
                backgroundColor: "#ffffff",
                fontSize: 14,
                textDecoration: "none",
                color: "#374151",
              }}
            >
              백엔드 /health 열기
            </a>
          </div>

          <p
            style={{
              marginTop: 10,
              fontSize: 11,
              color: "#9ca3af",
            }}
          >
            힌트: /ai/analyze-compare 라우트, GOOGLE_API_KEY, CORS 설정, 포트
            상태를 확인해 주세요.
          </p>
        </div>
      </div>
    )
  }

  const { left, right, analysis } = data || {}

  // ---------------- 실제 비교 화면 ----------------
  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(to bottom, #fdfaf5 0%, #f5eee3 40%, #f5f3ee 100%)",
        padding: "32px 16px 40px",
        boxSizing: "border-box",
        fontFamily:
          "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <Header onBack={() => navigate(-1)} />

        {/* 두 작품 카드 (좌/우) */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
            gap: 20,
            marginBottom: 24,
          }}
        >
          <ArtworkCard side="A" item={left} />
          <ArtworkCard side="B" item={right} />
        </div>

        {/* 비교 분석 텍스트 */}
        <section
          style={{
            marginTop: 8,
            borderRadius: 24,
            backgroundColor: "rgba(255,255,255,0.96)",
            boxShadow: "0 18px 45px rgba(15, 23, 42, 0.18)",
            border: "1px solid rgba(0,0,0,0.04)",
            padding: "20px 22px 22px",
          }}
        >
          <h2
            style={{
              margin: 0,
              marginBottom: 10,
              fontSize: 18,
              fontWeight: 600,
              color: "#111827",
            }}
          >
            두 작품 비교 해설
          </h2>
          <div
            style={{
              fontSize: 15,
              lineHeight: 1.75,
              color: "#374151",
              whiteSpace: "pre-wrap",
            }}
          >
            {analysis ||
              "두 작품에 대한 비교 설명을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."}
          </div>
        </section>

        <div
          style={{
            marginTop: 20,
            display: "flex",
            gap: 10,
            flexWrap: "wrap",
          }}
        >
          <button
            onClick={() => navigate(-1)}
            style={{
              padding: "8px 16px",
              borderRadius: 999,
              border: "none",
              backgroundColor: "#f3f4f6",
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            ← 전시장으로 돌아가기
          </button>

          <button
            onClick={retry}
            style={{
              padding: "8px 16px",
              borderRadius: 999,
              border: "1px solid #e5e7eb",
              backgroundColor: "#ffffff",
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            이 조합으로 다시 분석하기
          </button>

          <button
            onClick={handleOtherPair}
            disabled={!allIds.length || loadingAlt}
            style={{
              padding: "8px 16px",
              borderRadius: 999,
              border: "1px solid #e5e7eb",
              backgroundColor: "#ffffff",
              fontSize: 14,
              cursor: !allIds.length ? "not-allowed" : "pointer",
              opacity: !allIds.length ? 0.5 : 1,
            }}
          >
            {loadingAlt ? "다른 조합 찾는 중..." : "다른 조합으로 비교하기"}
          </button>
        </div>
      </div>
    </div>
  )
}

// ------------------------------------------------------
// 개별 작품 이미지 로딩 훅
// ------------------------------------------------------
function useArtworkImage(id) {
  const [url, setUrl] = useState(null)

  useEffect(() => {
    if (!id) return
    const controller = new AbortController()

    const run = async () => {
      try {
        const res = await fetch(`${API}/find_image/${encodeURIComponent(id)}`, {
          signal: controller.signal,
        })
        if (!res.ok) return
        const { url } = await res.json()
        setUrl(`${API}${url}`)
      } catch (e) {
        // console.error("이미지 로드 실패:", e)
      }
    }

    run()
    return () => controller.abort()
  }, [id])

  return url
}

// ------------------------------------------------------
// 상단 헤더
// ------------------------------------------------------
function Header({ onBack }) {
  return (
    <div
      style={{
        marginBottom: 24,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
      }}
    >
      <h1
        style={{
          margin: 0,
          fontSize: 22,
          fontWeight: 500,
          color: "#111827",
          fontFamily:
            "'Nanum Myeongjo', 'Apple SD Gothic Neo', 'Malgun Gothic', serif",
        }}
      >
        두 작품 비교
      </h1>
      <button
        onClick={onBack}
        style={{
          padding: "6px 12px",
          borderRadius: 999,
          border: "1px solid rgba(0,0,0,0.06)",
          backgroundColor: "rgba(255,255,255,0.9)",
          fontSize: 13,
          cursor: "pointer",
          color: "#4b5563",
        }}
      >
        ← 전시장으로 돌아가기
      </button>
    </div>
  )
}

// ------------------------------------------------------
// 개별 작품 카드
// ------------------------------------------------------
function ArtworkCard({ side, item }) {
  // 이미지 로딩 (id는 backend에서 card.setdefault("id", art_id) 해놨으니까 그대로 사용)
  const imgUrl = useArtworkImage(item?.id)

  // Detail.jsx와 동일한 규칙, card -> item 으로만 변경
  const title =
    item?.Description?.ArtTitle_kor ||
    item?.Description?.ArtTitle_eng ||
    item?.Data_Info?.ImageFileName ||
    item?.id ||
    "제목 없음"

  const artist =
    item?.Description?.ArtistName_kor ||
    item?.Description?.ArtistName_eng ||
    "작가 미상"

  const klass =
    item?.Description?.Class_kor ||
    item?.Description?.Class_eng ||
    ""

  const year = item?.Photo_Info?.PhotoDate || ""

  const material =
    item?.Description?.Material_kor ||
    item?.Description?.Material_eng ||
    ""

  const subtitleParts = [year, material].filter(Boolean).join(" · ")
  return (
    <div
      style={{
        borderRadius: 24,
        backgroundColor: "rgba(255,255,255,0.96)",
        boxShadow: "0 14px 35px rgba(15, 23, 42, 0.18)",
        border: "1px solid rgba(0,0,0,0.04)",
        padding: "16px 18px 18px",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          marginBottom: 10,
          fontSize: 12,
          color: "#9ca3af",
        }}
      >
        작품 {side}
      </div>

      {/* 이미지 영역 */}
      <div
        style={{
          width: "100%",
          aspectRatio: "4 / 3",
          borderRadius: 18,
          backgroundColor: "#ede9e4",
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 12,
        }}
      >
        {imgUrl ? (
          <img
            src={imgUrl}
            alt={title}
            style={{
              maxWidth: "100%",
              maxHeight: "100%",
              objectFit: "contain",
              display: "block",
            }}
          />
        ) : (
          <span style={{ color: "#9ca3af", fontSize: 13 }}>
            이미지를 불러올 수 없습니다.
          </span>
        )}
      </div>

      {/* 메타 정보 */}
      <div>
        <div
          style={{
            fontSize: 15,
            fontWeight: 600,
            color: "#111827",
            marginBottom: 2,
          }}
        >
          {title}
        </div>
        <div
          style={{
            fontSize: 14,
            color: "#4b5563",
            marginBottom: 4,
          }}
        >
          {artist}
        </div>
        {subtitleParts && (
          <div
            style={{
              fontSize: 13,
              color: "#6b7280",
            }}
          >
            {subtitleParts}
          </div>
        )}
      </div>
    </div>
  )
}