import React, { useEffect, useState } from "react"
import { useParams, useSearchParams, Link } from "react-router-dom"

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001"

const CATEGORY_MAP = {
  painting_json: "TL_01. 2D_02.회화(Json)",
  craft_json: "TL_01. 2D_04.공예(Json)",
  sculpture_json: "TL_01. 2D_06.조각(Json)",
}

export default function Detail() {
  const { id } = useParams()
  const [searchParams] = useSearchParams()
  const category = searchParams.get("category") || "painting_json"
  const mode = searchParams.get("mode") || "curate"

  // ✅ ArtworkGrid에서 넘겨준 variants (없을 수도 있음)
  const variantsParam = searchParams.get("variants")

  const [card, setCard] = useState(null)
  const [imgUrl, setImgUrl] = useState(null)

  const [imageVariants, setImageVariants] = useState([]) // [{id, url}, ...]
  const [mainImageIndex, setMainImageIndex] = useState(0)

  const [curation, setCuration] = useState("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const realFolder = CATEGORY_MAP[category] || category

  useEffect(() => {
    if (!id) return

    const run = async () => {
      setLoading(true)
      setError("")
      setImageVariants([])
      setMainImageIndex(0)

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
        setCard(cardJson)

        // 2) AI 큐레이션
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
        setCuration(curateData.curator_text || "")

        // 3) 이미지 로딩
        let candidateIds = []

        if (variantsParam) {
          // ✅ ArtworkGrid에서 넘겨준 id 리스트 우선 사용
          candidateIds = variantsParam
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        } else {
          // ✅ fallback: id 패턴으로 추측 (Welcome/Agent에서 바로 온 경우)
          const baseId = id.replace(/-\d+$/, "")
          for (let i = 1; i <= 10; i++) {
            candidateIds.push(`${baseId}-${i}`)
          }
        }

        // 항상 현재 id는 포함되도록
        if (!candidateIds.includes(id)) {
          candidateIds.unshift(id)
        }

        // 중복 제거
        candidateIds = Array.from(new Set(candidateIds))

        const variantResults = await Promise.all(
          candidateIds.map(async (cid) => {
            try {
              const res = await fetch(
                `${API}/find_image/${encodeURIComponent(cid)}`
              )
              if (!res.ok) return null
              const data = await res.json()
              return { id: cid, url: `${API}${data.url}` }
            } catch {
              return null
            }
          })
        )

        const validVariants = variantResults.filter(Boolean)

        if (validVariants.length > 0) {
          setImageVariants(validVariants)
          setImgUrl(validVariants[0].url)
        } else {
          // 그래도 아무 것도 못 찾으면 예전 방식으로 한 번 더 시도
          const imgRes = await fetch(
            `${API}/find_image/${encodeURIComponent(id)}`
          )
          if (imgRes.ok) {
            const imgData = await imgRes.json()
            const singleUrl = `${API}${imgData.url}`
            setImgUrl(singleUrl)
            setImageVariants([{ id, url: singleUrl }])
          } else {
            setImgUrl(null)
            setImageVariants([])
          }
        }
      } catch (e) {
        console.error(e)
        setError(
          e.message ||
            "작품 정보를 불러오는 중 문제가 발생했습니다. 다른 작품을 선택해 주세요."
        )
      } finally {
        setLoading(false)
      }
    }

    run()
  }, [id, realFolder, variantsParam])

  if (loading) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          fontFamily:
            "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        }}
      >
        <p style={{ fontSize: 18, marginBottom: 8 }}>
          AI 큐레이터가 전시를 준비 중입니다...
        </p>
        <p style={{ fontSize: 14, color: "#6b7280" }}>잠시만 기다려 주세요.</p>
      </div>
    )
  }

  if (error) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: 24,
          textAlign: "center",
          fontFamily:
            "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        }}
      >
        <p style={{ color: "#b91c1c", marginBottom: 12 }}>{error}</p>
        <Link
          to="/"
          style={{
            padding: "8px 16px",
            borderRadius: 999,
            border: "none",
            backgroundColor: "#e2b48a",
            color: "#332218",
            fontWeight: 600,
            textDecoration: "none",
          }}
        >
          처음 화면으로 돌아가기
        </Link>
      </div>
    )
  }

  // --------- 나머지 렌더링 부분은 네가 올린 코드 그대로 ---------
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
    card?.Description?.Material_kor ||
    card?.Description?.Material_eng ||
    ""

  const mainImage =
    imageVariants.length > 0
      ? imageVariants[Math.min(mainImageIndex, imageVariants.length - 1)]?.url
      : imgUrl

  // ----------------------------------------------------------
  // 렌더링
  // ----------------------------------------------------------
  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(to bottom, #fdfaf5 0%, #f5eee3 40%, #f5f3ee 100%)",
        padding: "32px 16px 40px",
        boxSizing: "border-box",
      }}
    >
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        {/* 상단 네비 */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: 24,
            gap: 12,
          }}
        >
          <Link
            to="/"
            style={{
              padding: "6px 12px",
              borderRadius: 999,
              border: "1px solid rgba(0,0,0,0.06)",
              backgroundColor: "rgba(255,255,255,0.9)",
              fontSize: 13,
              textDecoration: "none",
              color: "#4b5563",
            }}
          >
            ← 다른 전시 찾아보기
          </Link>
          <span style={{ fontSize: 13, color: "#9ca3af" }}>
            텍스트 해설 모드
          </span>
        </div>

        {/* 중앙 카드 */}
        <div
          style={{
            maxWidth: 900,
            margin: "0 auto",
            borderRadius: 28,
            backgroundColor: "rgba(255,255,255,0.96)",
            boxShadow: "0 18px 45px rgba(15, 23, 42, 0.22)",
            border: "1px solid rgba(0,0,0,0.04)",
            padding: "24px 26px 26px",
          }}
        >
          {/* 이미지 + 썸네일 영역 */}
          <div
            style={{
              width: "100%",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: "100%",
                maxWidth: 650,
                aspectRatio: "4 / 3",
                backgroundColor: "#ede9e4",
                borderRadius: 22,
                boxShadow: "0 14px 40px rgba(15,23,42,0.18)",
                overflow: "hidden",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: 16,
              }}
            >
              {mainImage ? (
                <img
                  src={mainImage}
                  alt={title}
                  style={{
                    maxWidth: "100%",
                    maxHeight: "60vh",
                    objectFit: "contain",
                  }}
                />
              ) : (
                <span style={{ color: "#6b7280", fontSize: 13 }}>
                  이미지를 불러올 수 없습니다.
                </span>
              )}
            </div>

            {/* ✅ 여러 이미지가 있을 때 썸네일 리스트 */}
            {imageVariants.length > 1 && (
              <div
                style={{
                  marginTop: 12,
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 8,
                  justifyContent: "center",
                }}
              >
                {imageVariants.map((v, idx) => (
                  <img
                    key={v.id}
                    src={v.url}
                    alt={`${title} - view ${idx + 1}`}
                    onClick={() => setMainImageIndex(idx)}
                    style={{
                      width: 64,
                      height: 64,
                      objectFit: "cover",
                      borderRadius: 10,
                      cursor: "pointer",
                      border:
                        idx === mainImageIndex
                          ? "2px solid #e2b48a"
                          : "1px solid rgba(0,0,0,0.08)",
                      opacity: idx === mainImageIndex ? 1 : 0.8,
                      boxShadow:
                        idx === mainImageIndex
                          ? "0 4px 10px rgba(0,0,0,0.18)"
                          : "none",
                    }}
                  />
                ))}
              </div>
            )}
          </div>

          {/* 텍스트 영역 */}
          <div>
            <h1
              style={{
                margin: 0,
                marginBottom: 8,
                fontSize: "clamp(22px, 3vw, 28px)",
                fontWeight: 500,
                color: "#1f2933",
                fontFamily:
                  "'Nanum Myeongjo', 'Apple SD Gothic Neo', 'Malgun Gothic', serif",
              }}
            >
              {title}
            </h1>

            <div style={{ marginBottom: 14, fontSize: 14, color: "#4b5563" }}>
              {artist && <span>{artist}</span>}
              {klass && (
                <>
                  {artist && " · "}
                  <span>{klass}</span>
                </>
              )}
              {(year || material) && (
                <div style={{ marginTop: 4, fontSize: 13, color: "#6b7280" }}>
                  {year && <span>{year}</span>}
                  {year && material && " · "}
                  {material && <span>{material}</span>}
                </div>
              )}
            </div>

            {/* 설명문 */}
            <div
              style={{
                padding: "14px 16px",
                borderRadius: 18,
                backgroundColor: "#f9fafb",
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
      </div>
    </div>
  )
}
