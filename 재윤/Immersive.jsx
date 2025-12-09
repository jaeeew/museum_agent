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

  // ---------- TTS / 오디오 상태 ----------
  const [segments, setSegments] = useState([]) // [{ text, step, paragraph }]
  const [activeIndex, setActiveIndex] = useState(-1)
  const [segmentTimings, setSegmentTimings] = useState([])

  const [audioUrl, setAudioUrl] = useState(null)
  const audioRef = useRef(null)

  const [audioLoading, setAudioLoading] = useState(false)
  const [audioReady, setAudioReady] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speechRate, setSpeechRate] = useState(1.0)

  // 이미지 크게 보여줄지 여부
  const [immersiveMode, setImmersiveMode] = useState(false)

  // 👉 추가: 도슨트(어두운 조명) 모드 여부
  const [docentMode, setDocentMode] = useState(false)

  // 음성 타입
  const [voiceType, setVoiceType] = useState("bright") // "bright" | "calm"
  const [voiceName, setVoiceName] = useState("ko-KR-Wavenet-A")

  // 화면 이동용
  const [tourStep, setTourStep] = useState(0) // 0~4
  const imageContainerRef = useRef(null)

  // 카테고리 → 실제 JSON 폴더
  const CATEGORY_MAP = {
    painting_json: "TL_01. 2D_02.회화(Json)",
    craft_json: "TL_01. 2D_04.공예(Json)",
    sculpture_json: "TL_01. 2D_06.조각(Json)",
  }

  const realFolder = CATEGORY_MAP[category] || category

  // 0. 음성 타입 → voiceName 매핑
  useEffect(() => {
    if (voiceType === "bright") {
      setVoiceName("ko-KR-Wavenet-A")
    } else {
      setVoiceName("ko-KR-Wavenet-C")
    }
  }, [voiceType])

  // 1. 카드 / 해설 / 이미지 로드
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

        // 2) AI 큐레이션 (몰입형 해설)
        const curateRes = await fetch(`${API}/curate/immersive`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id,
            category,
            card: cardJson,
          }),
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
          setError("서버 응답 오류: " + (e.message || "네트워크 오류"))
          setLoading(false)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    run()
    return () => {
      cancelled = true
    }
  }, [id, realFolder, category])

  // 2. 해설 텍스트 → (문단, 문장) 세그먼트 + 화면 step
  useEffect(() => {
    if (!curation) {
      setSegments([])
      setActiveIndex(-1)
      return
    }

    // 1) 먼저 문단 기준으로 나누기 (빈 줄 == 문단 구분)
    const paragraphs = curation
      .split(/\n\s*\n/) // "\n\n" 기준
      .map((p) => p.trim())
      .filter(Boolean)

    const newSegments = []

    paragraphs.forEach((paragraphText, pIndex) => {
      // 2) 각 문단을 문장 단위로 나누기
      const sentenceParts = paragraphText
        .split(/(?<=[\.!?])\s+/) // 마침표/느낌표/물음표 뒤 공백
        .map((s) => s.trim())
        .filter(Boolean)

      sentenceParts.forEach((text, idx) => {
        const hasLeft =
          text.includes("화면 왼쪽") ||
          text.includes("왼쪽") ||
          text.includes("좌측") ||
          text.includes("왼편")

        const hasRight =
          text.includes("화면 오른쪽") ||
          text.includes("오른쪽") ||
          text.includes("우측") ||
          text.includes("오른편")

        const hasCenter =
          text.includes("화면 가운데") ||
          text.includes("그림 가운데") ||
          text.includes("중앙") ||
          text.includes("한가운데")

        const hasTop =
          text.includes("위쪽") ||
          text.includes("윗부분") ||
          text.includes("상단")

        const hasBottom =
          text.includes("아래쪽") ||
          text.includes("아랫부분") ||
          text.includes("하단")

        let step = 0
        if (pIndex === 0) {
          // 1문단은 항상 전체 보기
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
        } else {
          step = 4 // 방향 언급 없으면 살짝 확대
        }

        newSegments.push({
          text,
          step,
          paragraph: pIndex,
        })
      })
    })

    setSegments(newSegments)
    setActiveIndex(-1)
    setSegmentTimings([])
  }, [curation])

  // 3. TTS 생성
  useEffect(() => {
    if (!curation) return

    let cancelled = false

    const run = async () => {
      try {
        setAudioLoading(true)
        setAudioReady(false)
        setIsPlaying(false)
        setAudioUrl(null)
        setActiveIndex(-1)
        setSegmentTimings([])

        const res = await fetch(`${API}/ai/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: curation,
            language_code: "ko-KR",
            voice_name: voiceName,
            speaking_rate: 1.0,
          }),
        })

        if (!res.ok) {
          const msg = await res.text().catch(() => "")
          throw new Error(msg || `TTS 생성 실패: ${res.status}`)
        }

        const json = await res.json()
        if (cancelled) return

        const url = `data:audio/mp3;base64,${json.audio_b64}`
        setAudioUrl(url)
        setAudioReady(true)
      } catch (e) {
        console.error(e)
        if (!cancelled) {
          setAudioUrl(null)
          setAudioReady(false)
        }
      } finally {
        if (!cancelled) setAudioLoading(false)
      }
    }

    run()
    return () => {
      cancelled = true
    }
  }, [curation, voiceName])

  // 4. 오디오 메타데이터 → 문장별 시간 추정
  const handleLoadedMetadata = () => {
    const el = audioRef.current
    if (!el || !segments.length) return

    const duration = el.duration
    if (!isFinite(duration) || duration <= 0) return

    const lengths = segments.map((s) => s.text.length || 1)
    const total = lengths.reduce((a, b) => a + b, 0)
    if (total <= 0) return

    let cum = 0
    const timings = segments.map((s, idx) => {
      const start = (cum / total) * duration
      cum += lengths[idx]
      const end = (cum / total) * duration
      return { start, end }
    })

    setSegmentTimings(timings)
  }

  // 오디오 진행 상황에 따라 activeIndex 업데이트
  const handleTimeUpdate = () => {
    const el = audioRef.current
    if (!el || !segmentTimings.length) return

    const t = el.currentTime
    const idx = segmentTimings.findIndex(
      (seg) => t >= seg.start && t < seg.end
    )

    if (idx !== -1 && idx !== activeIndex) {
      setActiveIndex(idx)
    }
  }

  const handleEnded = () => {
    setIsPlaying(false)
    setActiveIndex(-1)
    setImmersiveMode(false)
    setDocentMode(false)   // 👈 추가
  }

  // activeIndex → tourStep
  useEffect(() => {
    if (activeIndex < 0 || !segments.length) {
      setTourStep(0)
      return
    }
    setTourStep(segments[activeIndex].step)
  }, [activeIndex, segments])

  // 5. 재생 컨트롤
  const handlePlay = async () => {
    if (!audioRef.current || !audioUrl) return
    try {
      audioRef.current.playbackRate = speechRate
      await audioRef.current.play()
      setIsPlaying(true)
      setImmersiveMode(true)
      setDocentMode(true)
    } catch (e) {
      console.error(e)
    }
  }

  const handlePause = () => {
    if (!audioRef.current) return
    audioRef.current.pause()
    setIsPlaying(false)
  }

  const handleStop = () => {
    if (!audioRef.current) return
    audioRef.current.pause()
    audioRef.current.currentTime = 0
    setIsPlaying(false)
    setActiveIndex(-1)
    setImmersiveMode(false)
    setDocentMode(false)   // 👈 추가: 조명 모드 OFF
  }

  const handleChangeRate = (rate) => {
    setSpeechRate(rate)
    if (audioRef.current) {
      audioRef.current.playbackRate = rate
    }
  }

  // 6. 이미지 패닝 스타일 (줌을 조금만 쓰기)
  const getPanStyle = () => {
    switch (tourStep) {
      case 1:
        // 왼쪽/아래쪽 강조
        return {
          transform: "scale(1.25) translate(-8%, 4%)",
          transformOrigin: "left bottom",
        }
      case 2:
        // 가운데
        return {
          transform: "scale(1.35) translate(0%, 0%)",
          transformOrigin: "center center",
        }
      case 3:
        // 오른쪽/위쪽
        return {
          transform: "scale(1.25) translate(8%, -4%)",
          transformOrigin: "right top",
        }
      case 4:
        // 살짝 확대된 전체
        return {
          transform: "scale(1.1)",
          transformOrigin: "center center",
        }
      default:
        // 기본: 전체 보기
        return {
          transform: "scale(1.0)",
          transformOrigin: "center center",
        }
    }
  }

  // 7. 화면 렌더링
  if (loading) {
    return (
      <PageLayout wide={immersiveMode} docentMode={docentMode}>
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
          <p style={{ fontSize: 14, color: "#6b7280" }}>잠시만 기다려 주세요.</p>
        </div>
      </PageLayout>
    )
  }

  if (error) {
    return (
      <PageLayout wide={immersiveMode} docentMode={docentMode}>
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

  let tourStatusText = ""
  if (audioLoading) {
    tourStatusText = "TTS 음성을 준비하는 중입니다..."
  } else if (isPlaying) {
    tourStatusText = "현재 단계: 투어 진행 중"
  } else if (audioReady) {
    tourStatusText = "현재 단계: 일시정지됨"
  } else {
    tourStatusText = "현재 단계: 준비되지 않았습니다"
  }

  return (
    <PageLayout wide={immersiveMode}>
      {/* 숨겨진 오디오 요소 */}
      <audio
        ref={audioRef}
        src={audioUrl || undefined}
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        style={{ display: "none" }}
      />

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
            gridTemplateColumns: immersiveMode
              ? "minmax(0, 3.5fr) minmax(0, 0.7fr)"
              : "minmax(0, 2fr) minmax(0, 1.3fr)",
            gap: immersiveMode ? 30 : 24,
            alignItems: "flex-start",
            transition: "all 0.6s ease",
          }}
        >
          {/* 왼쪽: 이미지 영역 */}
          <div
            ref={imageContainerRef}
            style={{
              borderRadius: immersiveMode ? 34 : 28,
              backgroundColor: "#f5f3ef",
              boxShadow: immersiveMode
                ? "0 32px 90px rgba(15, 23, 42, 0.4)"
                : "0 18px 45px rgba(15, 23, 42, 0.22)",
              padding: immersiveMode ? 28 : 22,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflow: "hidden",
              transform: immersiveMode ? "scale(1.03)" : "scale(1.0)",
              transition:
                "transform 0.6s ease, padding 0.6s ease, border-radius 0.6s ease, box-shadow 0.6s ease",
            }}
          >
            <div
              style={{
                width: "100%",
                maxWidth: immersiveMode ? 1100 : 620,
                aspectRatio: immersiveMode ? "16 / 9" : "4 / 3",
                borderRadius: 22,
                overflow: "hidden",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: "#e5dfd8",
                transition: "all 0.6s ease",
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
                gap: 8,
                flexWrap: "wrap",
                marginBottom: 10,
              }}
            >
              <button
                onClick={handlePlay}
                disabled={!audioReady || audioLoading || !segments.length}
                style={{
                  padding: "8px 14px",
                  borderRadius: 999,
                  border: "none",
                  backgroundColor:
                    !audioReady || audioLoading || !segments.length
                      ? "#e5e7eb"
                      : "#f97316",
                  color:
                    !audioReady || audioLoading || !segments.length
                      ? "#9ca3af"
                      : "#fefaf4",
                  fontSize: 13,
                  cursor:
                    !audioReady || audioLoading || !segments.length
                      ? "not-allowed"
                      : "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <span>🎧 몰입형 투어 시작</span>
              </button>

              <button
                onClick={handlePause}
                disabled={!isPlaying}
                style={{
                  padding: "8px 14px",
                  borderRadius: 999,
                  border: "none",
                  backgroundColor: "#f3f4f6",
                  color: "#374151",
                  fontSize: 13,
                  cursor: !isPlaying ? "not-allowed" : "pointer",
                  opacity: !isPlaying ? 0.6 : 1,
                }}
              >
                ⏸ 투어 일시정지
              </button>

              <button
                onClick={handleStop}
                disabled={!audioReady}
                style={{
                  padding: "8px 14px",
                  borderRadius: 999,
                  border: "none",
                  backgroundColor: "#f3f4f6",
                  color: "#374151",
                  fontSize: 13,
                  cursor: !audioReady ? "not-allowed" : "pointer",
                  opacity: !audioReady ? 0.6 : 1,
                }}
              >
                ⏹ 투어 정지
              </button>
            </div>

            {/* 음성 타입 + 배속 컨트롤 */}
            <div
              style={{
                marginTop: 6,
                display: "flex",
                flexDirection: "column",
                gap: 8,
              }}
            >
              {/* 음성 타입 선택 */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  flexWrap: "wrap",
                }}
              >
                <span
                  style={{
                    fontSize: 12,
                    color: "#9ca3af",
                    marginRight: 4,
                  }}
                >
                  음성 톤
                </span>
                <button
                  onClick={() => setVoiceType("bright")}
                  disabled={audioLoading}
                  style={{
                    padding: "4px 10px",
                    borderRadius: 999,
                    border:
                      voiceType === "bright"
                        ? "1px solid #fb923c"
                        : "1px solid #e5e7eb",
                    backgroundColor:
                      voiceType === "bright" ? "#fff7ed" : "#ffffff",
                    fontSize: 12,
                    cursor: audioLoading ? "not-allowed" : "pointer",
                    color: "#374151",
                  }}
                >
                  여성
                </button>
                <button
                  onClick={() => setVoiceType("calm")}
                  disabled={audioLoading}
                  style={{
                    padding: "4px 10px",
                    borderRadius: 999,
                    border:
                      voiceType === "calm"
                        ? "1px solid #fb923c"
                        : "1px solid #e5e7eb",
                    backgroundColor:
                      voiceType === "calm" ? "#fff7ed" : "#ffffff",
                    fontSize: 12,
                    cursor: audioLoading ? "not-allowed" : "pointer",
                    color: "#374151",
                  }}
                >
                  남성
                </button>
                <span
                  style={{
                    fontSize: 11,
                    color: "#9ca3af",
                  }}
                >
                  (Google Wavenet 음성)
                </span>
              </div>

              {/* 재생 속도 */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  flexWrap: "wrap",
                }}
              >
                <span
                  style={{
                    fontSize: 12,
                    color: "#9ca3af",
                    marginRight: 4,
                  }}
                >
                  재생 속도
                </span>
                {[0.8, 1.0, 1.2, 1.5].map((rate) => (
                  <button
                    key={rate}
                    onClick={() => handleChangeRate(rate)}
                    style={{
                      padding: "4px 10px",
                      borderRadius: 999,
                      border:
                        speechRate === rate
                          ? "1px solid #fb923c"
                          : "1px solid #e5e7eb",
                      backgroundColor:
                        speechRate === rate ? "#fff7ed" : "#ffffff",
                      fontSize: 12,
                      cursor: "pointer",
                      color: "#374151",
                    }}
                  >
                    {rate.toFixed(1)}배
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 하단: 텍스트 해설 (현재 문장 하이라이트, 문단 유지) */}
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
            {segments.length ? (
              segments.map((seg, idx) => {
                const prev = idx > 0 ? segments[idx - 1] : null
                const isNewParagraph =
                  !prev || prev.paragraph !== seg.paragraph

                return (
                  <React.Fragment key={idx}>
                    {isNewParagraph && idx !== 0 && (
                      <>
                        <br />
                        <br />
                      </>
                    )}
                    <span
                      style={{
                        fontWeight: idx === activeIndex ? 700 : 400,
                        color: idx === activeIndex ? "#1d4ed8" : "#374151",
                      }}
                    >
                      {seg.text}
                    </span>{" "}
                  </React.Fragment>
                )
              })
            ) : (
              curation || "이 작품에 대한 설명을 불러오지 못했습니다."
            )}
          </div>
        </div>
      </div>
    </PageLayout>
  )
}

// 공통 레이아웃
function PageLayout({ children, wide = false, docentMode = false }) {
  const navigate = useNavigate()

  return (
    <div
    style={{
      minHeight: "100vh",
      background: docentMode
        ? "radial-gradient(circle at top, #111827 0%, #020617 50%, #000000 100%)"
        : "linear-gradient(to bottom, #fdfaf5 0%, #f5eee3 40%, #f5f3ee 100%)",
      padding: "24px 16px 40px",
      boxSizing: "border-box",
      transition: "background 0.5s ease",  // 👈 부드럽게 전환
    }}
    >
      <div
        style={{
          maxWidth: wide ? 1400 : 1180,
          margin: "0 auto",
          transition: "max-width 0.5s ease",
        }}
      >
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