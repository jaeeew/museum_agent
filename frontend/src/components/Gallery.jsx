// C:/Exhibit/curator_server/frontend/src/components/Detail.jsx

import React, { useEffect, useRef, useState } from "react"
import { useParams, useSearchParams, Link } from "react-router-dom"

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001"

export default function Detail() {
  const { id } = useParams()
  const [searchParams] = useSearchParams()
  const category = searchParams.get("category")
  const mode = searchParams.get("mode") || "curate"

  const [data, setData] = useState(null)
  const [imgUrl, setImgUrl] = useState(null)

  // 🧠 AI 설명문
  const [curation, setCuration] = useState("")
  const [loadingCuration, setLoadingCuration] = useState(false)
  const [showCuration, setShowCuration] = useState(true) // ← 기본적으로 펼쳐진 상태

  // 🔊 TTS 관련
  const voiceRef = useRef(null)
  const [ttsReady, setTtsReady] = useState(false)

  // -------------------- 데이터 로드 --------------------
  useEffect(() => {
    const loadDetail = async () => {
      try {
        const jsonUrl = `${API}/json_extracted/${category}/${id}.json`
        const res = await fetch(jsonUrl)
        const json = await res.json()
        setData(json)

        const imgRes = await fetch(`${API}/find_image/${id}`)
        if (imgRes.ok) {
          const { url } = await imgRes.json()
          setImgUrl(`${API}${url}`)
        }
      } catch (err) {
        console.error("❌ 상세정보 로드 실패:", err)
      }
    }
    loadDetail()
  }, [id, category])

  // -------------------- TTS 초기화 (Web Speech API) --------------------
  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      console.warn("이 브라우저는 Web Speech API를 지원하지 않습니다.")
      return
    }

    const pickKoreanVoice = () => {
      const voices = window.speechSynthesis.getVoices()
      if (!voices || voices.length === 0) return null

      const koVoices = voices.filter((v) => v.lang && v.lang.startsWith("ko"))
      if (koVoices.length === 0) return null

      const preferredKeywords = [
        "natural",
        "neural",
        "online",
        "cloud",
        "han",
        "heami",
        "sunhi",
        "google",
        "microsoft",
      ]

      for (const v of koVoices) {
        const nameLower = (v.name || "").toLowerCase()
        if (preferredKeywords.some((k) => nameLower.includes(k))) {
          return v
        }
      }
      return koVoices[0]
    }

    const handleVoicesChanged = () => {
      const v = pickKoreanVoice()
      voiceRef.current = v
      setTtsReady(true)
      console.log("선택된 한국어 음성:", v ? v.name : "기본 음성")
    }

    window.speechSynthesis.onvoiceschanged = handleVoicesChanged
    handleVoicesChanged()

    return () => {
      window.speechSynthesis.onvoiceschanged = null
    }
  }, [])

  const makeUtter = (text) => {
    const utter = new SpeechSynthesisUtterance(text)
    utter.lang = "ko-KR"
    utter.rate = 0.9
    utter.pitch = 0.95
    if (voiceRef.current) utter.voice = voiceRef.current
    return utter
  }

  const speakCuration = () => {
    if (!("speechSynthesis" in window)) {
      alert("이 브라우저는 음성 읽기를 지원하지 않습니다.")
      return
    }
    if (!curation) {
      alert("먼저 AI 설명문을 불러오는 중입니다.")
      return
    }
    window.speechSynthesis.cancel()
    const utter = makeUtter(curation)
    window.speechSynthesis.speak(utter)
  }

  const stopSpeech = () => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel()
    }
  }

  // -------------------- AI 설명문 요청 로직 --------------------
  const fetchCuration = async () => {
    if (!data || loadingCuration) return
    setLoadingCuration(true)
    try {
      const res = await fetch(`${API}/curate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, card: data }),
      })

      if (!res.ok) {
        const msg = await res.text().catch(() => "")
        throw new Error(`서버 오류 (${res.status}) ${msg}`)
      }

      const json = await res.json()
      setCuration(json.curator_text || "설명문 생성 실패")
    } catch (err) {
      console.error("❌ 설명문 생성 실패:", err)
      setCuration("AI 설명문을 불러오는 중 오류가 발생했습니다.")
    } finally {
      setLoadingCuration(false)
    }
  }

  // 화면 들어올 때 자동으로 한 번 호출
  useEffect(() => {
    if (data && !curation && !loadingCuration) {
      fetchCuration()
    }
  }, [data, id])

  // mode=tts 로 들어온 경우: 설명 생성 후 자동 재생
  useEffect(() => {
    if (mode === "tts" && curation && ttsReady) {
      speakCuration()
    }
  }, [mode, curation, ttsReady])

  // -------------------- 로딩 상태 --------------------
  if (!data) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#6b7280",
          fontFamily:
            "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        }}
      >
        📡 전시 정보를 불러오는 중입니다...
      </div>
    )
  }

  // -------------------- 데이터 정리 --------------------
  const desc = data.Description || {}
  const obj = data.Object_Info || {}
  const image = data.Image_Info || {}
  const datainfo = data.Data_Info || {}

  const titleKor = desc.ArtTitle_kor || data.title || "제목 없음"
  const artistKor = desc.ArtistName_kor || "작가 미상"
  const locationKor = desc.Location_kor || "-"
  const materialKor = desc.Material_kor || "-"
  const categoryKor = desc.Class_kor || obj.MiddleCategory || "-"

  // -------------------- 스타일 --------------------
  const pageStyle = {
    minHeight: "100vh",
    position: "relative",
    overflow: "hidden",
    fontFamily:
      "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    color: "#2f2832",
  }

  const overlayStyle = {
    position: "absolute",
    inset: 0,
    backgroundImage: `
      linear-gradient(
        to bottom,
        rgba(255,255,255,0.55),
        rgba(250,249,246,0.78),
        rgba(245,242,236,0.96)
      ),
      url("/museum-bg.jpg")
    `,
    backgroundSize: "cover",
    backgroundPosition: "center center",
    backgroundRepeat: "no-repeat",
    filter: "saturate(1.05)",
    zIndex: 0,
  }

  const layoutStyle = {
    position: "relative",
    zIndex: 1,
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",          // ← 가운데 정렬
    justifyContent: "flex-start",
    padding: "32px 16px 40px",
    boxSizing: "border-box",
  }

  const cardStyle = {
    width: "100%",
    maxWidth: "900px",
    margin: "0 auto",              // ← 가운데 정렬
    borderRadius: "26px",
    backgroundColor: "rgba(253,251,247,0.98)",
    boxShadow: "0 18px 45px rgba(15, 23, 42, 0.24)",
    border: "1px solid rgba(0,0,0,0.04)",
    padding: "24px 28px 28px",
    boxSizing: "border-box",
  }

  const sectionTitleStyle = {
    fontSize: "15px",
    fontWeight: 600,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "#b2794c",
    marginBottom: "6px",
  }

  // -------------------- 렌더링 --------------------
  return (
    <div style={pageStyle}>
      {/* 배경 */}
      <div style={overlayStyle} />

      {/* 콘텐츠 */}
      <div style={layoutStyle}>
        {/* 상단: 뒤로가기 */}
        <div
          style={{
            width: "100%",
            maxWidth: "1080px",
            marginBottom: "16px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Link
            to="/gallery"
            onClick={stopSpeech}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              fontSize: "13px",
              color: "#6b5b4b",
              textDecoration: "none",
              padding: "7px 12px",
              borderRadius: "999px",
              backgroundColor: "rgba(255,255,255,0.75)",
              border: "1px solid rgba(0,0,0,0.05)",
            }}
          >
            <span>←</span>
            <span>갤러리로 돌아가기</span>
          </Link>

          <span
            style={{
              fontSize: "12px",
              color: "#a08b77",
            }}
          >
            작품 ID: {id}
          </span>
        </div>

        {/* 메인 카드 */}
        <div style={cardStyle}>
          {/* 제목/작가 */}
          <div style={{ textAlign: "center", marginBottom: "18px" }}>
            <h1
              style={{
                margin: 0,
                fontSize: "clamp(24px, 3vw, 30px)",
                fontWeight: 500,
                letterSpacing: "0.04em",
                color: "#2b2118",
                fontFamily:
                  "'Nanum Myeongjo', 'Apple SD Gothic Neo', 'Malgun Gothic', serif",
              }}
            >
              {titleKor}
            </h1>
            <p
              style={{
                margin: "6px 0 0",
                fontSize: "14px",
                color: "#6b5b4b",
              }}
            >
              {artistKor}
            </p>
          </div>

          {/* 이미지 + 기본 정보 2단 레이아웃 */}
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              gap: "20px",
              alignItems: "flex-start",
              marginBottom: "18px",
            }}
          >
            {/* 이미지 영역 */}
            <div
  style={{
    flex: "0 0 45%",            // 52% → 45% 로 줄이기
    maxWidth: "420px",          // 이미지 영역 최대 폭 제한
    maxHeight: "420px",         // 이미지 영역 최대 높이 제한
    margin: "0 auto",           // 살짝 가운데 쪽으로
    borderRadius: "20px",
    overflow: "hidden",
    backgroundColor: "#f3f0ea",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  }}
>
  {imgUrl ? (
    <img
      src={imgUrl}
      alt={titleKor}
      style={{
        maxWidth: "100%",        // 부모 안에서만 확대
        maxHeight: "100%",
        width: "auto",
        height: "auto",
        objectFit: "contain",
        display: "block",
      }}
    />

              ) : (
                <div
                  style={{
                    padding: "40px 20px",
                    color: "#b0b0b0",
                    fontSize: "14px",
                  }}
                >
                  이미지를 불러올 수 없습니다.
                </div>
              )}
            </div>

            {/* 기본 정보 영역 (촬영정보 삭제됨) */}
            <div style={{ flex: "1 1 0", minWidth: 0 }}>
              <div
                style={{
                  backgroundColor: "#f7f4ef",
                  borderRadius: "18px",
                  padding: "14px 16px",
                }}
              >
                <div style={sectionTitleStyle}>기본 정보</div>
                <dl
                  style={{
                    margin: 0,
                    fontSize: "14px",
                    color: "#3f3a33",
                    lineHeight: 1.6,
                  }}
                >
                  <div>
                    <dt style={{ display: "inline", fontWeight: 600 }}>
                      분류&nbsp;·&nbsp;장르&nbsp;
                    </dt>
                    <dd style={{ display: "inline", margin: 0 }}>
                      {categoryKor}
                    </dd>
                  </div>
                  <div>
                    <dt style={{ display: "inline", fontWeight: 600 }}>
                      시대&nbsp;
                    </dt>
                    <dd style={{ display: "inline", margin: 0 }}>
                      {obj.MainCategory || "정보 없음"}
                    </dd>
                  </div>
                  <div>
                    <dt style={{ display: "inline", fontWeight: 600 }}>
                      소분류&nbsp;
                    </dt>
                    <dd style={{ display: "inline", margin: 0 }}>
                      {obj.SubCategory || "정보 없음"}
                    </dd>
                  </div>
                  <div>
                    <dt style={{ display: "inline", fontWeight: 600 }}>
                      재질&nbsp;
                    </dt>
                    <dd style={{ display: "inline", margin: 0 }}>
                      {materialKor}
                    </dd>
                  </div>
                  <div>
                    <dt style={{ display: "inline", fontWeight: 600 }}>
                      소재지&nbsp;
                    </dt>
                    <dd style={{ display: "inline", margin: 0 }}>
                      {locationKor}
                    </dd>
                  </div>
                  <div>
                    <dt style={{ display: "inline", fontWeight: 600 }}>
                      이미지 크기&nbsp;
                    </dt>
                    <dd style={{ display: "inline", margin: 0 }}>
                      {image.Width
                        ? `${image.Width} x ${image.Length} x ${
                            image.Height || "-"
                          }`
                        : "정보 없음"}
                    </dd>
                  </div>
                  <div>
                    <dt style={{ display: "inline", fontWeight: 600 }}>
                      파일명&nbsp;
                    </dt>
                    <dd style={{ display: "inline", margin: 0 }}>
                      {datainfo.ImageFileName || id}
                    </dd>
                  </div>
                  <div>
                    <dt style={{ display: "inline", fontWeight: 600 }}>
                      형식&nbsp;
                    </dt>
                    <dd style={{ display: "inline", margin: 0 }}>
                      {datainfo.SourceDataExtension || "jpg"}
                    </dd>
                  </div>
                </dl>
              </div>
            </div>
          </div>

          {/* 원본 메타 설명 */}
          <div
            style={{
              backgroundColor: "#f8f5f0",
              borderRadius: "18px",
              padding: "14px 16px",
              border: "1px solid rgba(0,0,0,0.03)",
              marginBottom: "16px",
            }}
          >
            <div style={sectionTitleStyle}>작품 설명</div>
            <p
              style={{
                margin: 0,
                fontSize: "14px",
                color: "#494037",
                lineHeight: 1.7,
              }}
            >
              {desc.ArtTitle_kor && desc.ArtTitle_eng ? (
                <>
                  <strong>{desc.ArtTitle_kor}</strong>
                  <br />
                  <span
                    style={{
                      fontSize: "13px",
                      color: "#8c7a68",
                      fontStyle: "italic",
                    }}
                  >
                    {desc.ArtTitle_eng}
                  </span>
                </>
              ) : (
                "작품 설명 없음"
              )}
            </p>
          </div>

          {/* 🧠 AI 설명문 + 🔊 TTS 버튼 */}
          <div style={{ marginTop: "10px" }}>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "8px",
                marginBottom: "10px",
              }}
            >
              <button
                onClick={fetchCuration}
                style={{
                  flex: "1 1 200px",
                  padding: "10px 14px",
                  borderRadius: "999px",
                  border: "none",
                  backgroundColor: "#e2b48a",
                  color: "#342218",
                  fontWeight: 600,
                  fontSize: "14px",
                  cursor: "pointer",
                  boxShadow:
                    "0 10px 25px rgba(145, 104, 61, 0.35)",
                  whiteSpace: "nowrap",
                }}
              >
                {loadingCuration ? "🧠 설명문 다시 생성 중..." : "🧠 AI 설명문 다시 생성"}
              </button>

              <button
                onClick={speakCuration}
                disabled={!ttsReady}
                style={{
                  padding: "10px 14px",
                  borderRadius: "999px",
                  border: "none",
                  fontSize: "13px",
                  fontWeight: 600,
                  cursor: ttsReady ? "pointer" : "not-allowed",
                  backgroundColor: ttsReady ? "#d29bb8" : "#e5e7eb",
                  color: ttsReady ? "#fff" : "#9ca3af",
                  boxShadow: ttsReady
                    ? "0 8px 20px rgba(180,90,130,0.35)"
                    : "none",
                }}
              >
                🔊 AI 설명 듣기
              </button>

              <button
                onClick={stopSpeech}
                style={{
                  padding: "10px 14px",
                  borderRadius: "999px",
                  border: "none",
                  fontSize: "13px",
                  fontWeight: 500,
                  cursor: "pointer",
                  backgroundColor: "#f3f4f6",
                  color: "#4b5563",
                }}
              >
                ⏹ 멈추기
              </button>
            </div>

            {showCuration && (
              <div
                style={{
                  marginTop: "4px",
                  backgroundColor: "#f7f4ef",
                  borderRadius: "16px",
                  padding: "14px 16px",
                  border: "1px solid rgba(0,0,0,0.03)",
                  fontSize: "14px",
                  color: "#3f3a33",
                  lineHeight: 1.7,
                  whiteSpace: "pre-wrap",
                }}
              >
                {loadingCuration && !curation ? (
                  <span style={{ color: "#2563eb" }}>
                    ⌛ 설명문을 준비하고 있습니다...
                  </span>
                ) : (
                  curation || "아직 생성된 설명문이 없습니다."
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
