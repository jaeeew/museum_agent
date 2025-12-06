// C:/Exhibit/curator_server/frontend/src/components/Immersive.jsx
import React, { useEffect, useMemo, useRef, useState } from "react"
import { useLocation, useNavigate } from "react-router-dom"

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

  const [card, setCard] = useState(null)
  const [imgUrl, setImgUrl] = useState(null)
  const [curation, setCuration] = useState("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  // ---------- TTS 관련 상태 ----------
  const [ttsReady, setTtsReady] = useState(false)
  const [tourRunning, setTourRunning] = useState(false)
  const [tourStep, setTourStep] = useState(0)

  const voiceRef = useRef(null)
  const utteranceRef = useRef(null)

  // 화면 이동용 ref
  const imageContainerRef = useRef(null)

  // 카테고리 → 폴더 매핑 (Detail.jsx와 동일하게 맞춰 줌)
  const CATEGORY_MAP = {
    painting_json: "TL_01. 2D_02.회화(Json)",
    craft_json: "TL_01. 2D_04.공예(Json)",
    sculpture_json: "TL_01. 2D_06.조각(Json)",
  }

  const realFolder = CATEGORY_MAP[category] || category

  // ==============================
  // 1. 작품 / 이미지 / 해설문 불러오기
  // ==============================
  useEffect(() => {
    if (!id) return

    let cancelled = false

    const run = async () => {
      setLoading(true)
      setError("")
      try {
        // 1) 카드 JSON
        const jsonUrl = `${API}/json_extracted/${encodeURIComponent(
          realFolder
        )}/${encodeURIComponent(id)}.json`

        const cardRes = await fetch(jsonUrl)
        if (!cardRes.ok)
          throw new Error(`카드 JSON 로드 실패: ${cardRes.status}`)

        const cardJson = await cardRes.json()
        if (!cardJson.id) cardJson.id = id
        if (cancelled) return
        setCard(cardJson)

        // 2) AI 큐레이션 (Detail에서 쓰는 것과 동일한 /curate 엔드포인트)
        const curateRes = await fetch(`${API}/curate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id, card: cardJson }),
        })
        if (!curateRes.ok) {
          const msg = await curateRes.text().catch(() => "")
          throw new Error(msg || `큐레이션 생성 실패: ${curateRes.status}`)
        }
        const curateData = await curateRes.json()
        if (cancelled) return
        setCuration(curateData.curator_text || "")

        // 3) 이미지 URL
        const imgRes = await fetch(
          `${API}/find_image/${encodeURIComponent(id)}`
        )
        if (!imgRes.ok) {
          throw new Error(`이미지 검색 실패: ${imgRes.status}`)
        }
        const imgData = await imgRes.json()
        if (cancelled) return
        setImgUrl(`${API}${imgData.url}`)
      } catch (e) {
        console.error(e)
        if (!cancelled) {
          setError(
            e.message ||
              "작품 정보를 불러오는 중 문제가 발생했습니다. 다른 작품을 선택해 주세요."
          )
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    run()

    return () => {
      cancelled = true
    }
  }, [id, realFolder])

  // ==============================
  // 2. 브라우저 TTS 초기화
  // ==============================
  useEffect(() => {
    if (typeof window === "undefined") return
    if (!("speechSynthesis" in window)) {
      setTtsReady(false)
      return
    }

    const initVoices = () => {
      const voices = window.speechSynthesis.getVoices()
      if (!voices || voices.length === 0) return

      const korVoice =
        voices.find((v) => v.lang.startsWith("ko")) ||
        voices.find((v) => v.lang.startsWith("en")) ||
        voices[0]

      voiceRef.current = korVoice
      setTtsReady(true)
    }

    initVoices()
    window.speechSynthesis.onvoiceschanged = initVoices

    return () => {
      window.speechSynthesis.onvoiceschanged = null
    }
  }, [])

  const stopTTS = () => {
    if (!("speechSynthesis" in window)) return
    window.speechSynthesis.cancel()
    utteranceRef.current = null
  }

  useEffect(() => {
    return () => {
      stopTTS()
    }
  }, [])

  // ==============================
  // 3. 문장 → 화면 포인트 매핑 로직
  //    (여기서 "왼쪽/오른쪽/중앙/글씨/꽃" 같은 단어를 보고
  //     자동으로 어디를 확대할지 결정함)
  // ==============================
  const segments = useMemo(() => {
    if (!curation) return []

    // 기본 문장 분리
    const rawSentences = curation
      .split(/(?<=[\.!?])\s+|\n+/) // 마침표 + 공백, 줄바꿈 기준
      .map((s) => s.trim())
      .filter(Boolean)

    const mapped = rawSentences.map((text, idx) => {
      const t = text.toLowerCase()

      // 한글 키워드는 소문자 처리 필요 X 이지만 그냥 같이 사용
      const hasLeft =
        text.includes("왼쪽") || text.includes("좌측") || text.includes("왼편")
      const hasRight =
        text.includes("오른쪽") || text.includes("우측") || text.includes("오른편")
      const hasCenter =
        text.includes("가운데") || text.includes("중앙") || text.includes("한가운데")
      const hasTop = text.includes("위쪽") || text.includes("윗부분") || text.includes("상단")
      const hasBottom = text.includes("아래") || text.includes("하단") || text.includes("아랫부분")

      const mentionsFlower =
        text.includes("꽃") ||
        text.includes("꽃잎") ||
        text.includes("꽃송이") ||
        text.includes("꽃이")
      const mentionsBranch =
        text.includes("가지") ||
        text.includes("나뭇가지") ||
        text.includes("줄기")
      const mentionsText =
        text.includes("글씨") ||
        text.includes("문장") ||
        text.includes("서예") ||
        text.includes("글자")

      // 0: 전체, 1: 왼쪽/아래, 2: 중앙, 3: 오른쪽/위, 4: 전체 약간 줌
      let step = 0

      // 첫 문장은 작품 전체 소개용
      if (idx === 0) {
        step = 0
      } else if (hasLeft && hasBottom) {
        step = 1
      } else if (hasLeft) {
        step = 1
      } else if (hasRight && hasTop) {
        step = 3
      } else if (hasRight) {
        step = 3
      } else if (hasCenter) {
        step = 2
      } else if (hasTop) {
        step = 3
      } else if (hasBottom) {
        step = 1
      } else if (mentionsText) {
        // 서예/글씨는 보통 오른쪽 or 중앙
        step = 3
      } else if (mentionsFlower || mentionsBranch) {
        // 꽃/나뭇가지는 화면 왼쪽/아래에 있는 경우가 많아서 1번에 매핑
        step = 1
      } else {
        // 특별한 키워드가 없으면 살짝 줌인 정도
        step = 4
      }

      return { text, step }
    })

    return mapped
  }, [curation])

  // ==============================
  // 4. 화면 포커싱 스타일
  // ==============================
  const getPanStyle = () => {
    switch (tourStep) {
      case 1:
        return {
          transform: "scale(1.4) translate(-10%, 5%)",
          transformOrigin: "left bottom",
        }
      case 2:
        return {
          transform: "scale(1.4) translate(0%, 0%)",
          transformOrigin: "center center",
        }
      case 3:
        return {
          transform: "scale(1.4) translate(10%, -5%)",
          transformOrigin: "right top",
        }
      case 4:
        return {
          transform: "scale(1.2)",
          transformOrigin: "center center",
        }
      default:
        return {
          transform: "scale(1.0)",
          transformOrigin: "center center",
        }
    }
  }

  // ==============================
  // 5. 몰입형 투어 시작 (문장 단위 TTS + 화면 이동)
  // ==============================
  const startImmersiveTour = () => {
    if (!ttsReady || !segments.length) return

    stopTTS()
    setTourRunning(true)

    let index = 0

    const playSegment = () => {
      if (index >= segments.length) {
        setTourRunning(false)
        setTourStep(0)
        return
      }

      const { text, step } = segments[index]

      // 화면 포인트 먼저 바꾸고
      setTourStep(step)

      // 그다음 해당 문장을 읽어 줌
      const utter = new SpeechSynthesisUtterance(text)
      if (voiceRef.current) utter.voice = voiceRef.current
      utter.rate = 1.0
      utter.pitch = 1.0

      utter.onend = () => {
        index += 1
        playSegment()
      }

      utteranceRef.current = utter
      window.speechSynthesis.speak(utter)
    }

    playSegment()
  }

  const stopImmersiveTour = () => {
    stopTTS()
    setTourRunning(false)
    setTourStep(0)
  }

  // ==============================
  // 6. 화면 렌더링
  // ==============================
  if (loading) {
    return (
      <PageLayout>
        <div
          style={{
            minHeight: "60vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <p style={{ fontSize: 18, marginBottom: 8 }}>
            몰입형 작품 감상을 준비하고 있어요...
          </p>
          <p style={{ fontSize: 14, color: "#6b7280" }}>
            잠시만 기다려 주세요.
          </p>
        </div>
      </PageLayout>
    )
  }

  if (error) {
    return (
      <PageLayout>
        <div
          style={{
            minHeight: "60vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: 24,
            textAlign: "center",
          }}
        >
          <p style={{ color: "#b91c1c", marginBottom: 12 }}>{error}</p>
          <button
            onClick={() => navigate(-1)}
            style={{
              padding: "8px 16px",
              borderRadius: 999,
              border: "none",
              backgroundColor: "#e2b48a",
              color: "#332218",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            전시장으로 돌아가기
          </button>
        </div>
      </PageLayout>
    )
  }

  const title =
    card?.Description?.ArtTitle_kor ||
    card?.Description?.ArtTitle_eng ||
    card?.Data_Info?.ImageFileName ||
    id

  const artist =
    card?.Description?.ArtistName_kor || card?.Description?.ArtistName_eng || ""
  const klass =
    card?.Description?.Class_kor || card?.Description?.Class_eng || ""
  const year = card?.Photo_Info?.PhotoDate || ""
  const material =
    card?.Description?.Material_kor || card?.Description?.Material_eng || ""

  const tourStatusText = tourRunning
    ? "현재 단계: 투어 진행 중"
    : "현재 단계: 투어 종료"

  return (
    <PageLayout>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        {/* 상단 제목 영역 */}
        <div>
          <h2
            style={{
              margin: 0,
              marginBottom: 6,
              fontSize: 22,
              fontWeight: 500,
              color: "#1f2933",
            }}
          >
            몰입형 작품 감상
          </h2>
          <div style={{ fontSize: 13, color: "#6b7280" }}>
            화면의 움직임과 음성 해설을 따라가며, 실제 전시장에서 작품을 둘러보는 것
            같은 경험을 제공합니다.
          </div>
        </div>

        {/* 메인 영역 (이미지 + 카드) */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1.3fr)",
            gap: 24,
            alignItems: "flex-start",
          }}
        >
          {/* 왼쪽: 이미지 영역 */}
          <div
            ref={imageContainerRef}
            style={{
              borderRadius: 28,
              backgroundColor: "#f5f3ef",
              boxShadow: "0 18px 45px rgba(15, 23, 42, 0.22)",
              padding: 22,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: "100%",
                maxWidth: 650,
                aspectRatio: "4 / 3",
                borderRadius: 22,
                overflow: "hidden",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: "#e5dfd8",
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
                    transition: "transform 1.2s ease-in-out",
                    ...getPanStyle(),
                  }}
                />
              ) : (
                <span style={{ color: "#6b7280", fontSize: 13 }}>
                  이미지를 불러올 수 없습니다.
                </span>
              )}
            </div>
          </div>

          {/* 오른쪽: 작품 정보 + 투어 컨트롤 카드 */}
          <div
            style={{
              borderRadius: 24,
              backgroundColor: "rgba(255,255,255,0.98)",
              boxShadow: "0 16px 40px rgba(15, 23, 42, 0.18)",
              padding: 22,
            }}
          >
            <div style={{ marginBottom: 14 }}>
              <div
                style={{
                  fontSize: 13,
                  color: "#9ca3af",
                  marginBottom: 4,
                }}
              >
                몰입형 해설 대상 작품
              </div>
              <h3
                style={{
                  margin: 0,
                  marginBottom: 4,
                  fontSize: 20,
                  fontWeight: 600,
                  color: "#111827",
                }}
              >
                {title}
              </h3>
              <div style={{ fontSize: 14, color: "#4b5563" }}>
                {artist && <span>{artist}</span>}
                {klass && (
                  <>
                    {artist && " · "}
                    <span>{klass}</span>
                  </>
                )}
                {(year || material) && (
                  <div
                    style={{ marginTop: 4, fontSize: 13, color: "#6b7280" }}
                  >
                    {year && <span>{year}</span>}
                    {year && material && " · "}
                    {material && <span>{material}</span>}
                  </div>
                )}
              </div>
            </div>

            <div
              style={{
                padding: "12px 14px",
                borderRadius: 16,
                backgroundColor: "#f9fafb",
                fontSize: 13,
                color: "#4b5563",
                marginBottom: 14,
                lineHeight: 1.6,
              }}
            >
              <div style={{ marginBottom: 6 }}>
                음성 해설을 들으면서 화면 속 작품을 따라가 보세요. 실제 미술관에서
                작품 앞에 서서, 시선이 옮겨 다니는 느낌을 그대로 옮겨 놓았습니다.
              </div>
              <div style={{ color: "#9ca3af", fontSize: 12 }}>
                {tourStatusText}
              </div>
            </div>

            {/* 버튼 영역 */}
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
                disabled={!ttsReady || !segments.length}
                style={{
                  padding: "8px 14px",
                  borderRadius: 999,
                  border: "none",
                  backgroundColor:
                    !ttsReady || !segments.length ? "#e5e7eb" : "#f97316",
                  color:
                    !ttsReady || !segments.length ? "#9ca3af" : "#fefaf4",
                  fontSize: 13,
                  cursor:
                    !ttsReady || !segments.length ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <span>🎧 몰입형 투어 시작</span>
              </button>

              <button
                onClick={stopImmersiveTour}
                style={{
                  padding: "8px 14px",
                  borderRadius: 999,
                  border: "none",
                  backgroundColor: "#f3f4f6",
                  color: "#374151",
                  fontSize: 13,
                  cursor: "pointer",
                }}
              >
                ⏹ 투어 / 음성 정지
              </button>
            </div>

            {!ttsReady && (
              <div
                style={{
                  fontSize: 12,
                  color: "#b91c1c",
                  marginTop: 4,
                }}
              >
                이 브라우저에서는 음성 합성이 지원되지 않을 수 있습니다. 다른
                브라우저에서 다시 시도해 주세요.
              </div>
            )}
          </div>
        </div>

        {/* 하단: 텍스트 해설 전문 */}
        <div
          style={{
            marginTop: 10,
            borderRadius: 24,
            backgroundColor: "rgba(255,255,255,0.96)",
            boxShadow: "0 14px 32px rgba(15, 23, 42, 0.16)",
            padding: 20,
          }}
        >
          <div
            style={{
              fontSize: 13,
              color: "#9ca3af",
              marginBottom: 6,
            }}
          >
            AI 큐레이터의 해설
          </div>
          <div
            style={{
              fontSize: 15,
              lineHeight: 1.7,
              color: "#374151",
              whiteSpace: "pre-wrap",
            }}
          >
            {curation || "이 작품에 대한 설명을 불러오지 못했습니다."}
          </div>
        </div>
      </div>
    </PageLayout>
  )
}

// ==============================
// 공통 레이아웃 컴포넌트
// ==============================
function PageLayout({ children }) {
  const navigate = useNavigate()

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(to bottom, #fdfaf5 0%, #f5eee3 40%, #f5f3ee 100%)",
        padding: "24px 16px 40px",
        boxSizing: "border-box",
      }}
    >
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 18,
          }}
        >
          <button
            onClick={() => navigate(-1)}
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
