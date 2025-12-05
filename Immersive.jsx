// C:/Exhibit/curator_server/frontend/src/components/Immersive.jsx

import React, { useEffect, useMemo, useRef, useState } from "react"
import { useLocation, useNavigate, Link } from "react-router-dom"

const API = import.meta.env?.VITE_API_BASE || "http://127.0.0.1:8001"

export default function Immersive() {
  const location = useLocation()
  const navigate = useNavigate()

  const params = useMemo(
    () => new URLSearchParams(location.search),
    [location.search]
  )

  const id = params.get("id") || ""
  const category = params.get("category") || "painting_json"

  // 디버그용 로그 (필요 없으면 지워도 됨)
  useEffect(() => {
    console.log("[Immersive] location.search:", location.search)
    console.log("[Immersive] id param:", id)
    console.log("[Immersive] category param:", category)
  }, [location.search, id, category])

  const [card, setCard] = useState(null)
  const [imgUrl, setImgUrl] = useState(null)
  const [curation, setCuration] = useState("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  // ----- TTS 관련 -----
  const voiceRef = useRef(null)
  const [ttsReady, setTtsReady] = useState(false)
  const utteranceRef = useRef(null)

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return
    }

    const synth = window.speechSynthesis

    const handleVoices = () => {
      const voices = synth.getVoices()
      const koVoice =
        voices.find(v => v.lang?.startsWith("ko")) || voices[0] || null
      voiceRef.current = koVoice
      setTtsReady(true)
    }

    handleVoices()
    synth.addEventListener("voiceschanged", handleVoices)

    return () => {
      synth.removeEventListener("voiceschanged", handleVoices)
    }
  }, [])

  const speakCuration = text => {
    if (!ttsReady || !text) return
    const synth = window.speechSynthesis
    synth.cancel()

    const u = new SpeechSynthesisUtterance(text)
    if (voiceRef.current) u.voice = voiceRef.current
    u.rate = 1.0
    u.pitch = 1.0
    utteranceRef.current = u
    synth.speak(u)
  }

  const stopTTS = () => {
    if (!("speechSynthesis" in window)) return
    window.speechSynthesis.cancel()
  }

  // ----- 이미지 프레임/투어 상태 -----
  const frameRef = useRef(null)
  const imgRef = useRef(null)
  const [tourStep, setTourStep] = useState(0)
  const [tourRunning, setTourRunning] = useState(false)
  const tourTimersRef = useRef([])

  const imageTransform = useMemo(() => {
    let scale = 1.0
    let originX = "50%"
    let originY = "50%"

    if (tourStep === 0) {
      scale = 1.15
      originX = "50%"
      originY = "50%"
    } else if (tourStep === 1) {
      scale = 2.0
      originX = "20%"
      originY = "80%"
    } else if (tourStep === 2) {
      scale = 2.2
      originX = "50%"
      originY = "40%"
    } else if (tourStep === 3) {
      scale = 2.0
      originX = "80%"
      originY = "50%"
    } else if (tourStep === 4) {
      scale = 1.1
      originX = "50%"
      originY = "50%"
    } else {
      scale = 1.0
      originX = "50%"
      originY = "50%"
    }

    return { scale, originX, originY }
  }, [tourStep])

  useEffect(() => {
    const img = imgRef.current
    const frame = frameRef.current
    if (!img || !frame) return

    img.style.transformOrigin = `${imageTransform.originX} ${imageTransform.originY}`
    img.style.transform = `scale(${imageTransform.scale})`

    const rect = frame.getBoundingClientRect()
    const top = rect.top + window.scrollY
    window.scrollTo({ top: top - 40, behavior: "smooth" })
  }, [imageTransform])

  const clearTourTimers = () => {
    tourTimersRef.current.forEach(t => clearTimeout(t))
    tourTimersRef.current = []
  }

  const startImmersiveTour = () => {
    if (!curation) return
    clearTourTimers()
    stopTTS()

    setTourRunning(true)
    setTourStep(0)
    speakCuration(curation)

    const stepTimes = [0, 6000, 12000, 18000, 24000, 30000]

    stepTimes.forEach((ms, idx) => {
      const t = setTimeout(() => {
        setTourStep(idx)
        if (idx === stepTimes.length - 1) {
          setTourRunning(false)
        }
      }, ms)
      tourTimersRef.current.push(t)
    })
  }

  const stopImmersiveTour = () => {
    setTourRunning(false)
    clearTourTimers()
    stopTTS()
    setTourStep(5)
  }

  // ----- 카드 & 이미지 & 큐레이션 로드 -----
  useEffect(() => {
    if (!id) {
      setError("작품 ID가 없습니다.")
      setLoading(false)
      return
    }

    const controller = new AbortController()

    const run = async () => {
      try {
        setLoading(true)
        setError("")

        const folder =
          category === "painting_json"
            ? "TL_01. 2D_02.회화(Json)"
            : category === "craft_json"
            ? "TL_01. 2D_04.공예(Json)"
            : "TL_01. 2D_06.조각(Json)"

        const cardRes = await fetch(
          `${API}/json_extracted/${encodeURIComponent(folder)}/${encodeURIComponent(id)}.json`,
          { signal: controller.signal }
        )

        console.log(
          "[Immersive] JSON URL:",
          `${API}/json_extracted/${encodeURIComponent(folder)}/${encodeURIComponent(id)}.json`
        )

        if (!cardRes.ok) {
          throw new Error(`카드 정보를 불러오지 못했습니다. (${cardRes.status})`)
        }

        const cardJson = await cardRes.json()
        setCard(cardJson)

        const imgRes = await fetch(
          `${API}/find_image/${encodeURIComponent(id)}`,
          { signal: controller.signal }
        )
        if (imgRes.ok) {
          const { url } = await imgRes.json()
          setImgUrl(`${API}${url}`)
        }

        const curateRes = await fetch(`${API}/curate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id, card: cardJson }),
          signal: controller.signal,
        })
        if (!curateRes.ok) {
          const t = await curateRes.text().catch(() => "")
          throw new Error(t || "큐레이션 텍스트를 불러오지 못했습니다.")
        }
        const curateJson = await curateRes.json()
        setCuration(curateJson.curator_text || "")
      } catch (e) {
        if (e.name !== "AbortError") {
          setError(
            e.message ||
              "몰입형 투어 데이터를 불러오는 중 오류가 발생했습니다."
          )
        }
      } finally {
        setLoading(false)
      }
    }

    run()
    return () => {
      controller.abort()
      clearTourTimers()
      stopTTS()
    }
  }, [id, category])

  const title =
    card?.Description?.ArtTitle_kor ||
    card?.Description?.ArtTitle_eng ||
    card?.Data_Info?.ImageFileName ||
    id

  const artist =
    card?.Description?.ArtistName_kor ||
    card?.Description?.ArtistName_eng ||
    ""

  const year = card?.Photo_Info?.PhotoDate || ""
  const material =
    card?.Description?.Material_kor ||
    card?.Description?.Material_eng ||
    ""

  // ----- 유사 작품 추천 상태 -----
  const [similarItems, setSimilarItems] = useState([])
  const [similarLoading, setSimilarLoading] = useState(false)

  // 유사도 점수 예쁘게 포맷팅
  const formatSimilarity = score => {
    if (score == null) return ""
    // chroma cosine distance면 0에 가까울수록 유사 → 1 - d 로 전환
    const s = 1 - score
    const pct = Math.round(s * 100)
    return `${pct}%`
  }

  // 유사 작품 추천 로드
  useEffect(() => {
    if (!id) return

    const loadSimilar = async () => {
      try {
        setSimilarLoading(true)
        setSimilarItems([])

        const url = `${API}/similar_images?id=${encodeURIComponent(
          id
        )}&category=${encodeURIComponent(category)}`

        console.log("[Immersive] similar_images:", url)

        const res = await fetch(url)
        if (!res.ok) {
          console.warn("similar_images res not ok:", res.status)
          return
        }
        const data = await res.json()
        const items = data.items || []
        setSimilarItems(items)
      } catch (e) {
        console.warn("similar_images error:", e)
      } finally {
        setSimilarLoading(false)
      }
    }

    loadSimilar()
  }, [id, category])

  if (loading) {
    return (
      <PageLayout onBack={() => navigate(-1)}>
        <div
          style={{
            marginTop: 80,
            textAlign: "center",
            color: "#6b7280",
          }}
        >
          🎧 몰입형 작품 투어를 준비하고 있습니다...
        </div>
      </PageLayout>
    )
  }

  if (error) {
    return (
      <PageLayout onBack={() => navigate(-1)}>
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
      </PageLayout>
    )
  }

  return (
    <PageLayout onBack={() => navigate(-1)}>
      <h1
        style={{
          margin: 0,
          fontSize: 22,
          fontWeight: 500,
          color: "#111827",
          fontFamily:
            "'Nanum Myeongjo', 'Apple SD Gothic Neo', 'Malgun Gothic', serif",
          marginBottom: 16,
        }}
      >
        몰입형 작품 감상
      </h1>

      {/* 상단: 이미지 + 투어 컨트롤 */}
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)",
          gap: 24,
          alignItems: "flex-start",
          marginBottom: 24,
        }}
      >
        <div
          ref={frameRef}
          style={{
            borderRadius: 28,
            backgroundColor: "rgba(255,255,255,0.96)",
            boxShadow: "0 18px 45px rgba(15,23,42,0.16)",
            border: "1px solid rgba(0,0,0,0.04)",
            padding: 18,
          }}
        >
          <div
            style={{
              borderRadius: 22,
              backgroundColor: "#ede9e4",
              overflow: "hidden",
              width: "100%",
              aspectRatio: "4 / 3",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {imgUrl ? (
              <img
                ref={imgRef}
                id="art-image"
                src={imgUrl}
                alt={title}
                style={{
                  maxWidth: "100%",
                  maxHeight: "100%",
                  objectFit: "contain",
                  transition:
                    "transform 2.2s ease-out, transform-origin 2.2s ease-out",
                }}
              />
            ) : (
              <span style={{ color: "#9ca3af", fontSize: 14 }}>
                이미지를 불러올 수 없습니다.
              </span>
            )}
          </div>
        </div>

        <div
          style={{
            borderRadius: 24,
            backgroundColor: "rgba(255,255,255,0.96)",
            boxShadow: "0 12px 30px rgba(15,23,42,0.12)",
            border: "1px solid rgba(0,0,0,0.03)",
            padding: "18px 18px 16px",
          }}
        >
          <div
            style={{
              fontSize: 18,
              fontWeight: 600,
              color: "#111827",
              marginBottom: 4,
            }}
          >
            {title}
          </div>
          <div
            style={{
              fontSize: 15,
              color: "#4b5563",
              marginBottom: 4,
            }}
          >
            {artist || "작가 미상"}
          </div>
          <div
            style={{
              fontSize: 13,
              color: "#6b7280",
              marginBottom: 12,
            }}
          >
            {[year, material].filter(Boolean).join(" · ")}
          </div>

          <div
            style={{
              marginBottom: 12,
              fontSize: 13,
              color: "#6b7280",
              lineHeight: 1.6,
            }}
          >
            음성 해설을 들으면서 화면 속 작품을 따라가 보세요.  
            실제 미술관에서 작품 앞에 서서, 시선이 옮겨 다니는 느낌을
            그대로 옮겨 놓았습니다.
          </div>

          <div
            style={{
              display: "flex",
              gap: 10,
              flexWrap: "wrap",
              marginBottom: 10,
            }}
          >
            <button
              onClick={startImmersiveTour}
              disabled={!ttsReady || !curation}
              style={{
                padding: "8px 16px",
                borderRadius: 999,
                border: "none",
                backgroundColor: "#f97316",
                color: "#fff",
                fontSize: 14,
                cursor: "pointer",
                opacity: !ttsReady || !curation ? 0.6 : 1,
              }}
            >
              {tourRunning ? "투어 다시 시작" : "몰입형 투어 시작"}
            </button>
            <button
              onClick={stopImmersiveTour}
              style={{
                padding: "8px 14px",
                borderRadius: 999,
                border: "1px solid #e5e7eb",
                backgroundColor: "#fff",
                fontSize: 13,
                cursor: "pointer",
              }}
            >
              투어 / 음성 정지
            </button>
          </div>

          <div
            style={{
              fontSize: 12,
              color: "#9ca3af",
            }}
          >
            현재 단계:{" "}
            {tourStep <= 0
              ? "전체 보기"
              : tourStep === 1
              ? "왼쪽 아래 디테일"
              : tourStep === 2
              ? "중앙 초점"
              : tourStep === 3
              ? "오른쪽 요소"
              : tourStep === 4
              ? "다시 전체 구도"
              : "투어 종료"}
          </div>
        </div>
      </section>

      {/* 큐레이션 텍스트 */}
      {curation && (
        <section
          style={{
            borderRadius: 24,
            backgroundColor: "rgba(255,255,255,0.96)",
            boxShadow: "0 12px 30px rgba(15,23,42,0.12)",
            border: "1px solid rgba(0,0,0,0.03)",
            padding: "18px 20px 20px",
            fontSize: 15,
            color: "#374151",
            lineHeight: 1.7,
            whiteSpace: "pre-wrap",
            marginBottom: 24,
          }}
        >
          {curation}
        </section>
      )}

      {/* ⭐ 맨 아래: 유사 작품 추천 카드 섹션 */}
      <section style={{ marginTop: 8, marginBottom: 24 }}>
        <h2
          style={{
            fontSize: 18,
            fontWeight: 600,
            color: "#1f2937",
            marginBottom: 12,
            fontFamily: "'Nanum Myeongjo', serif",
          }}
        >
          비슷한 작품도 감상해 보세요
        </h2>

        {similarLoading ? (
          <p style={{ color: "#6b7280", fontSize: 14 }}>
            유사 작품을 불러오는 중입니다...
          </p>
        ) : similarItems.length === 0 ? (
          <p style={{ color: "#9ca3af", fontSize: 14 }}>
            유사한 작품을 찾지 못했습니다.
          </p>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
              gap: 16,
              marginTop: 4,
            }}
          >
            {similarItems.map(item => {
              const simText = formatSimilarity(item.score)
              const targetCategory = item.category || category

              return (
                <Link
                  key={item.id}
                  to={`/detail/${item.id}?category=${encodeURIComponent(
                    targetCategory
                  )}`}
                  style={{ textDecoration: "none", color: "inherit" }}
                >
                  <div
                    style={{
                      position: "relative",
                      borderRadius: 14,
                      backgroundColor: "white",
                      boxShadow: "0 6px 14px rgba(15,23,42,0.12)",
                      padding: 10,
                      cursor: "pointer",
                      transition:
                        "transform 0.15s ease-out, box-shadow 0.15s ease-out",
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.transform = "translateY(-2px)"
                      e.currentTarget.style.boxShadow =
                        "0 10px 20px rgba(15,23,42,0.18)"
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.transform = "translateY(0)"
                      e.currentTarget.style.boxShadow =
                        "0 6px 14px rgba(15,23,42,0.12)"
                    }}
                  >
                    {item.image_path ? (
                      <img
                        src={`${API}${item.image_path}`}
                        alt={item.title}
                        style={{
                          width: "100%",
                          height: 110,
                          objectFit: "cover",
                          borderRadius: 10,
                          marginBottom: 8,
                          backgroundColor: "#f3f4f6",
                        }}
                      />
                    ) : (
                      <div
                        style={{
                          width: "100%",
                          height: 110,
                          borderRadius: 10,
                          marginBottom: 8,
                          background:
                            "repeating-linear-gradient(45deg, #f3f4f6, #f3f4f6 6px, #e5e7eb 6px, #e5e7eb 12px)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 11,
                          color: "#9ca3af",
                        }}
                      >
                        이미지 없음
                      </div>
                    )}

                    <div
                      style={{
                        fontSize: 13,
                        fontWeight: 500,
                        color: "#111827",
                        marginBottom: 2,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {item.title || "제목 미상"}
                    </div>
                    <div
                      style={{
                        fontSize: 12,
                        color: "#6b7280",
                        marginBottom: 2,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {item.artist || "작가 미상"}
                    </div>
                    {simText && (
                      <div
                        style={{
                          fontSize: 11,
                          color: "#2563eb",
                          marginTop: 2,
                        }}
                      >
                        유사도 {simText}
                      </div>
                    )}
                  </div>
                </Link>
              )
            })}
          </div>
        )}
      </section>
    </PageLayout>
  )
}

function PageLayout({ children, onBack }) {
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
        <div
          style={{
            marginBottom: 20,
            display: "flex",
            justifyContent: "flex-end",
          }}
        >
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

        {children}
      </div>
    </div>
  )
}
