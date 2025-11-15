import React, { useEffect, useMemo, useState } from "react"
import { useLocation, useNavigate } from "react-router-dom"
// Vite 사용 시 .env에서 VITE_API_BASE 지정 가능, 없으면 로컬 8000
const API = import.meta.env?.VITE_API_BASE || "http://localhost:8000"

export default function Compare() {
  const location = useLocation()
  const navigate = useNavigate()
  const params = useMemo(() => new URLSearchParams(location.search), [location.search])

  const idsParam = params.get("ids") || ""
  const category = params.get("category") || "painting_json"
  const ids = idsParam.split(",").map(s => decodeURIComponent(s.trim())).filter(Boolean)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [data, setData] = useState(null)   // { left, right, analysis }

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (ids.length !== 2) {
        setError("비교할 작품 ID 2개가 필요합니다")
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
          signal: controller.signal
        })
        if (!res.ok) {
          const t = await res.text().catch(() => "")
          throw new Error(t || `분석 요청 실패 (HTTP ${res.status})`)
        }
        const json = await res.json()
        setData(json)
      } catch (e) {
        if (e.name !== "AbortError") setError(e.message || "에러가 발생했습니다")
      } finally {
        setLoading(false)
      }
    }
    run()
    return () => controller.abort()
  }, [idsParam, category])

  const retry = () => {
    // 쿼리를 유지한 채로 강제 재실행
    navigate(0)
  }

  if (loading) {
    return (
      <div className="p-8">
        <Header onBack={() => navigate(-1)} />
        <div className="text-center text-gray-500 mt-20 animate-pulse">
          🔎 제미나이가 두 작품을 비교 분석 중...
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-8">
        <Header onBack={() => navigate(-1)} />
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700 whitespace-pre-wrap">
          {error}
        </div>
        <div className="mt-4 flex gap-3">
          <button onClick={retry} className="px-4 py-2 rounded-xl bg-gray-100 hover:bg-gray-200">다시 시도</button>
          <a
            href={`${API}/health`}
            target="_blank"
            rel="noreferrer"
            className="px-4 py-2 rounded-xl bg-white border hover:bg-gray-50"
          >
            백엔드 /health 열기
          </a>
        </div>
        <p className="text-xs text-gray-500 mt-3">
          힌트: .env의 GOOGLE_GENAI_API_KEY, CORS, /ai/analyze-compare 라우트, 8000 포트 확인
        </p>
      </div>
    )
  }

  const { left, right, analysis } = data || {}

  return (
    <div className="p-8">
      <Header onBack={() => navigate(-1)} />

      {/* 좌우 썸네일 + 메타 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <ArtworkCard side="A" item={left} category={category} />
        <ArtworkCard side="B" item={right} category={category} />
      </div>

      {/* 분석 결과 */}
      <section className="prose max-w-none">
        <h2 className="text-xl font-bold mb-3">분석 결과</h2>
        {/* 기본은 프리텍스트. 마크다운 렌더링 원하면 react-markdown 붙여줄 수 있어 */}
        <pre className="whitespace-pre-wrap leading-7 text-gray-800">{analysis}</pre>
      </section>

      <div className="mt-8 flex gap-3">
        <button onClick={() => navigate(-1)} className="px-4 py-2 rounded-xl bg-gray-100 hover:bg-gray-200">
          ← 갤러리로
        </button>
        <button onClick={retry} className="px-4 py-2 rounded-xl bg-white border hover:bg-gray-50">
          새로 분석하기
        </button>
      </div>
    </div>
  )
}

function Header({ onBack }) {
  return (
    <div className="flex items-center justify-between mb-6">
      <h1 className="text-2xl font-bold">🆚 두 작품 비교</h1>
      <button onClick={onBack} className="px-4 py-2 rounded-xl bg-gray-100 hover:bg-gray-200">
        ← 갤러리로
      </button>
    </div>
  )
}

function ArtworkCard({ side, item, category }) {
  return (
    <div className="border rounded-2xl p-4">
      {item?.image_url ? (
        <img src={item.image_url} alt={item?.title || side} className="w-full h-80 object-cover rounded-xl mb-3" />
      ) : (
        <div className="w-full h-80 bg-gray-100 rounded-xl mb-3 flex items-center justify-center text-gray-400">
          이미지 없음
        </div>
      )}
      <div className="text-sm text-gray-700">
        <div className="font-semibold">{item?.title || "제목 없음"}</div>
        <div className="text-gray-500">{item?.artist || "작가 미상"}</div>
        <div className="text-gray-500">{item?.year || ""}</div>
        <div className="text-gray-500">{[item?.material, item?.size].filter(Boolean).join(" · ")}</div>
        <div className="text-xs text-gray-400 mt-1">ID: {item?.id} · {category}</div>
      </div>
    </div>
  )
}
