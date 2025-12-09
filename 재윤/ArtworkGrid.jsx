// C:/Exhibit/curator_server/frontend/src/components/ArtworkGrid.jsx

import React, { useEffect, useState } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001"

const CATEGORY_MAP = {
   "painting_json": "TL_01. 2D_02.회화(Json)",
    "craft_json":    "TL_01. 2D_04.공예(Json)",
    "sculpture_json": "TL_01. 2D_06.조각(Json)",
}

// 한 페이지에서 JSON 파일을 "많이" 가져와서
// 중복 묶기를 해도 그룹 카드가 15~20개 이상 나오도록 여유 있게 잡음
const JSON_FILES_PER_PAGE = 135  // 이전: 30

export default function ArtworkGrid() {
  const [searchParams] = useSearchParams()
  const initialCategory = searchParams.get("category") || "painting_json"

  const [category] = useState(initialCategory) // 화면에서 바꾸는 버튼 없으므로 고정
  const [allFiles, setAllFiles] = useState([])
  const [page, setPage] = useState(1)
  const [pageItems, setPageItems] = useState([]) // 중복 제거된 카드 데이터
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [expandedId, setExpandedId] = useState(null)

  // 💡 모드: 기본 / 비교
  const [mode, setMode] = useState("default") // "default" | "compare"
  const [selected, setSelected] = useState([]) // 선택된 작품 id들

  const navigate = useNavigate()

  const realFolder = CATEGORY_MAP[category] || category

  // 모드 바뀌면 선택 초기화
  useEffect(() => {
    setSelected([])
  }, [mode])

  // 1️⃣ 카테고리별 JSON 파일 목록 로드
  useEffect(() => {
    const loadList = async () => {
      try {
        setLoading(true)
        setError("")
        setExpandedId(null)

        const res = await fetch(`${API}/json_list/${category}`)
        if (!res.ok) {
          throw new Error(`파일 목록 로드 실패: ${res.status}`)
        }
        const list = await res.json()
        setAllFiles(list)
        setPage(1)
      } catch (err) {
        console.error("❌ 목록 로드 실패:", err)
        setAllFiles([])
        setError(
          err.message || "작품 목록을 불러오는 중 문제가 발생했습니다."
        )
      } finally {
        setLoading(false)
      }
    }

    loadList()
  }, [category])

  // 2️⃣ 현재 페이지의 JSON들 로드해서 카드 데이터 만들기 (+중복 제거)
  useEffect(() => {
    const loadPage = async () => {
      if (!allFiles.length) {
        setPageItems([])
        return
      }

      setLoading(true)
      setError("")
      setExpandedId(null)

      try {
        // ✅ 한 페이지에서 JSON을 많이(120개) 가져와서,
        // 중복 묶기 이후에도 그룹 카드가 여러 줄로 꽉 차게 보이도록
        const start = (page - 1) * JSON_FILES_PER_PAGE
        const currentFiles = allFiles.slice(start, start + JSON_FILES_PER_PAGE)

        const rawItems = await Promise.all(
          currentFiles.map(async (file) => {
            const jsonUrl = `${API}/json_extracted/${encodeURIComponent(
              realFolder
            )}/${encodeURIComponent(file)}`

            const res = await fetch(jsonUrl)
            if (!res.ok) {
              throw new Error(`JSON 로드 실패: ${res.status} (${file})`)
            }
            const json = await res.json()

            const desc = json.Description || {}
            const title =
              desc.ArtTitle_kor ||
              desc.ArtTitle_eng ||
              json.title ||
              "제목 없음"
            const artist =
              desc.ArtistName_kor ||
              desc.ArtistName_eng ||
              json.artist ||
              "작가 미상"

            const prefix = file.replace(/\.[^/.]+$/, "")

            // 이미지 URL
            let imgUrl = null
            try {
              const imgRes = await fetch(
                `${API}/find_image/${encodeURIComponent(prefix)}`
              )
              if (imgRes.ok) {
                const imgData = await imgRes.json()
                imgUrl = `${API}${imgData.url}`
              }
            } catch (e) {
              console.warn("이미지 찾기 실패:", e)
            }

            return {
              id: prefix,
              img: imgUrl,
              meta: { title, artist, category },
            }
          })
        )

        // 🔹 같은 제목+작가인 작품들을 하나로 묶기 (중복 제거)
        const map = new Map()
        rawItems.forEach((item) => {
          const key = `${item.meta.title}__${item.meta.artist}`
          if (!map.has(key)) {
            map.set(key, {
              ...item,
              variants: [{ id: item.id, img: item.img }],
            })
          } else {
            const group = map.get(key)
            group.variants.push({ id: item.id, img: item.img })
          }
        })

        const grouped = Array.from(map.values())
        setPageItems(grouped)
      } catch (err) {
        console.error("❌ 페이지 로드 실패:", err)
        setPageItems([])
        setError(
          err.message || "작품 데이터를 불러오는 중 문제가 발생했습니다."
        )
      } finally {
        setLoading(false)
      }
    }

    loadPage()
  }, [allFiles, page, realFolder, category])

  const totalPages = Math.ceil(allFiles.length / JSON_FILES_PER_PAGE) || 1

  // 🔘 카드 클릭 동작
  const handleCardClick = (item) => {
    // 🧭 기본 모드: 바로 상세 페이지로 이동
    if (mode === "default") {
      navigate(`/detail/${encodeURIComponent(item.id)}?category=${category}`)
      return
    }

    // 🟣 비교 모드: 최대 2개 선택
    if (mode === "compare") {
      setSelected((prev) => {
        const exists = prev.includes(item.id)
        if (exists) return prev.filter((id) => id !== item.id)
        if (prev.length >= 2) return prev
        return [...prev, item.id]
      })
      return
    }
  }

  // 🔍 개별 작품 상세 보기 버튼 (어떤 모드에서도 동작)
  const handleOpenDetail = (item) => {
    const firstId = item.variants?.[0]?.id || item.id

    const variantIds = (item.variants || [{ id: item.id }])
      .map((v) => v.id)
      .filter(Boolean)
      .join(",")

    navigate(
      `/detail/${encodeURIComponent(firstId)}?category=${category}` +
        (variantIds ? `&variants=${encodeURIComponent(variantIds)}` : "")
    )
  }

  // 🆚 두 작품 비교하기 실행
  const handleCompare = () => {
    if (selected.length !== 2) return
    const [a, b] = selected
    navigate(
      `/compare?ids=${encodeURIComponent(a)},${encodeURIComponent(
        b
      )}&category=${category}`
    )
  }

  // 모드별 안내 텍스트
  const modeHint = (() => {
    if (mode === "compare") {
      return "비교 모드: 작품 카드 두 개를 클릭해서 선택한 뒤, 아래의 ‘두 작품 비교하기’ 버튼을 눌러 주세요."
    }
    return "기본 모드: 작품 카드를 클릭하면 상세 해설 화면으로 이동합니다."
  })()

  const prettyCategoryName = (() => {
    if (category === "painting_json") return "갤러리"
    if (category === "craft_json") return "공예 갤러리"
    if (category === "sculpture_json") return "조각 갤러리"
    return category
  })()

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
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        {/* 상단 헤더 */}
        <header style={{ textAlign: "center", marginBottom: 26 }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "4px 12px",
              borderRadius: 999,
              border: "1px solid rgba(148,163,184,0.4)",
              fontSize: 11,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "#6b7280",
              backgroundColor: "rgba(255,255,255,0.9)",
              marginBottom: 10,
            }}
          >
            Curated Collection
          </div>
          <h1
            style={{
              margin: 0,
              fontSize: 24,
              fontWeight: 600,
              color: "#111827",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              gap: 8,
              fontFamily:
                "'Apple SD Gothic Neo', 'Nanum Gothic', system-ui, sans-serif",
            }}
          >
            <span role="img" aria-label="palette">
              🎨
            </span>
            {prettyCategoryName}
          </h1>
          <p
            style={{
              marginTop: 8,
              marginBottom: 14,
              fontSize: 13,
              color: "#6b7280",
            }}
          >
            작품을 클릭해 상세 해설을 보거나, 두 작품을 나란히 비교해 보세요.
          </p>

          {/* 모드 전환 버튼 */}
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: 8,
              marginTop: 6,
            }}
          >
            <ModeButton
              label="기본 모드"
              active={mode === "default"}
              onClick={() => setMode("default")}
            />
            <ModeButton
              label="⚖ 비교 모드"
              active={mode === "compare"}
              onClick={() => setMode("compare")}
            />
          </div>

          <p
            style={{
              marginTop: 10,
              fontSize: 12,
              color: "#9ca3af",
            }}
          >
            {modeHint}
          </p>
        </header>

        {/* 에러 */}
        {error && (
          <div
            style={{
              textAlign: "center",
              color: "#b91c1c",
              marginBottom: 16,
              fontSize: 13,
            }}
          >
            {error}
          </div>
        )}

        {/* 로딩 / 카드 그리드 */}
        {loading ? (
          <div
            style={{
              textAlign: "center",
              color: "#6b7280",
              marginTop: 40,
              fontSize: 14,
            }}
          >
            📡 작품 데이터를 불러오는 중입니다...
          </div>
        ) : (
          <>
            {/* 카드 그리드: 화면 가로폭 꽉 채우기 (4~5열 반응형) */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(220px, 1fr))",
                gap: "24px",
                justifyItems: "stretch",
              }}
            >
              {pageItems.map((item, idx) => {
                const isSelected = selected.includes(item.id)

                return (
                  <div
                    key={item.id ?? idx}
                    style={{
                      position: "relative",
                      borderRadius: 18,
                      border: isSelected
                        ? "1.5px solid #fb923c"
                        : "1px solid rgba(0,0,0,0.06)",
                      boxShadow: isSelected
                        ? "0 10px 26px rgba(248,113,113,0.35)"
                        : "0 6px 18px rgba(15,23,42,0.12)",
                      padding: 12,
                      backgroundColor: "#ffffff",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "stretch",
                      transition:
                        "transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease",
                      cursor: "pointer",
                    }}
                    onClick={() => handleCardClick(item)}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = "translateY(-3px)"
                      if (!isSelected) {
                        e.currentTarget.style.boxShadow =
                          "0 10px 24px rgba(15,23,42,0.18)"
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = "translateY(0)"
                      e.currentTarget.style.boxShadow = isSelected
                        ? "0 10px 26px rgba(248,113,113,0.35)"
                        : "0 6px 18px rgba(15,23,42,0.12)"
                    }}
                  >
                    {/* 선택 표시 뱃지 */}
                    {isSelected && (
                      <div
                        style={{
                          position: "absolute",
                          top: 8,
                          right: 8,
                          width: 22,
                          height: 22,
                          borderRadius: "999px",
                          background:
                            "linear-gradient(135deg, #fb923c, #f97316)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          color: "#fff",
                          fontSize: 13,
                          fontWeight: 700,
                          boxShadow:
                            "0 4px 10px rgba(248,113,113,0.55)",
                        }}
                      >
                        ✓
                      </div>
                    )}

                    {/* 대표 이미지 + 제목/작가 영역 */}
                    <div style={{ width: "100%", textAlign: "center" }}>
                      {/* 이미지 래퍼: 셀 폭에 맞춰 꽉 차게 */}
                      <div
                        style={{
                          width: "100%",
                          borderRadius: 18,
                          overflow: "hidden",
                          backgroundColor: "#e5e7eb",
                          aspectRatio: "1 / 1", // 정사각형
                          marginBottom: 10,
                        }}
                      >
                        <img
                          src={item.img || item.variants?.[0]?.img}
                          alt={item.meta.title}
                          style={{
                            width: "100%",
                            height: "100%",
                            objectFit: "cover",
                            display: "block",
                          }}
                        />
                      </div>

                      <p
                        style={{
                          fontSize: 13,
                          fontWeight: 600,
                          color: "#374151",
                          textAlign: "center",
                          marginBottom: 4,
                        }}
                      >
                        {item.meta.title}
                      </p>
                      <p
                        style={{
                          fontSize: 12,
                          color: "#6b7280",
                          textAlign: "center",
                        }}
                      >
                        {item.meta.artist}
                      </p>
                    </div>

                    {/* 펼쳐진 경우: 해당 작품의 모든 이미지 썸네일 */}
                    {expandedId === item.id &&
                      item.variants &&
                      item.variants.length > 1 && (
                        <div
                          style={{
                            marginTop: 10,
                            paddingTop: 8,
                            borderTop: "1px solid #e5e7eb",
                            display: "flex",
                            flexWrap: "wrap",
                            gap: 6,
                            justifyContent: "center",
                          }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          {item.variants.map((v, i) => (
                            <img
                              key={v.id ?? i}
                              src={v.img || item.img}
                              alt={`${item.meta.title} - view ${i + 1}`}
                              style={{
                                width: 60,
                                height: 60,
                                objectFit: "cover",
                                borderRadius: 8,
                              }}
                            />
                          ))}
                        </div>
                      )}

                    {/* 카드 하단 버튼들 */}
                    <div
                      style={{
                        marginTop: 10,
                        display: "flex",
                        gap: 6,
                        justifyContent: "center",
                        width: "100%",
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        type="button"
                        onClick={() => handleOpenDetail(item)}
                        style={{
                          flex: "0 0 auto",
                          padding: "6px 10px",
                          borderRadius: 999,
                          border: "1px solid #e5e7eb",
                          backgroundColor: "#f9fafb",
                          fontSize: 11,
                          color: "#4b5563",
                          cursor: "pointer",
                        }}
                      >
                        상세 보기
                      </button>
                      {item.variants && item.variants.length > 1 && (
                        <button
                          type="button"
                          onClick={() =>
                            setExpandedId((prev) =>
                              prev === item.id ? null : item.id
                            )
                          }
                          style={{
                            flex: "0 0 auto",
                            padding: "6px 10px",
                            borderRadius: 999,
                            border: "1px solid #e5e7eb",
                            backgroundColor: "#ffffff",
                            fontSize: 11,
                            color: "#6b7280",
                            cursor: "pointer",
                          }}
                        >
                          {expandedId === item.id ? "이미지 접기" : "다른 이미지"}
                        </button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            {/* 비교 모드 하단 액션 바 */}
            {mode === "compare" && (
              <div
                style={{
                  marginTop: 18,
                  padding: "10px 14px",
                  borderRadius: 18,
                  backgroundColor: "rgba(255,255,255,0.85)",
                  border: "1px solid rgba(209,213,219,0.9)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 10,
                  fontSize: 13,
                  color: "#4b5563",
                }}
              >
                <div>
                  선택된 작품&nbsp;
                  <strong>{selected.length}</strong>/2
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button
                    onClick={handleCompare}
                    disabled={selected.length !== 2}
                    style={{
                      padding: "7px 14px",
                      borderRadius: 999,
                      border: "none",
                      backgroundColor:
                        selected.length === 2 ? "#f97316" : "#e5e7eb",
                      color:
                        selected.length === 2 ? "#fff7ed" : "#9ca3af",
                      fontSize: 13,
                      cursor:
                        selected.length === 2 ? "pointer" : "not-allowed",
                    }}
                  >
                    두 작품 비교하기
                  </button>
                </div>
              </div>
            )}

            {/* 페이지네이션 */}
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                marginTop: 24,
                gap: 12,
              }}
            >
              <button
                onClick={() => setPage((p) => Math.max(p - 1, 1))}
                style={{
                  padding: "8px 14px",
                  borderRadius: 999,
                  border: "1px solid #e5e7eb",
                  backgroundColor: "#f9fafb",
                  fontSize: 13,
                  cursor: page === 1 ? "not-allowed" : "pointer",
                  color: page === 1 ? "#9ca3af" : "#374151",
                }}
                disabled={page === 1}
              >
                ◀ 이전
              </button>
              <span
                style={{
                  fontSize: 13,
                  color: "#6b7280",
                  minWidth: 80,
                  textAlign: "center",
                }}
              >
                {page} / {totalPages}
              </span>
              <button
                onClick={() =>
                  setPage((p) => Math.min(p + 1, totalPages))
                }
                style={{
                  padding: "8px 14px",
                  borderRadius: 999,
                  border: "1px solid #e5e7eb",
                  backgroundColor: "#f9fafb",
                  fontSize: 13,
                  cursor:
                    page === totalPages ? "not-allowed" : "pointer",
                  color:
                    page === totalPages ? "#9ca3af" : "#374151",
                }}
                disabled={page === totalPages}
              >
                다음 ▶
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// 상단 모드 전환 버튼 컴포넌트
// ─────────────────────────────────────────────
function ModeButton({ label, active, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: "6px 14px",
        borderRadius: 999,
        border: active
          ? "1px solid #fb923c"
          : "1px solid rgba(209,213,219,0.9)",
        backgroundColor: active ? "#fff7ed" : "rgba(255,255,255,0.9)",
        fontSize: 12,
        color: active ? "#ea580c" : "#4b5563",
        cursor: "pointer",
        boxShadow: active ? "0 4px 12px rgba(248,113,113,0.35)" : "none",
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
      }}
    >
      {label}
    </button>
  )
}