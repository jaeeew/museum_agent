// C:/Exhibit/curator_server/frontend/src/components/ArtworkGrid.jsx

import React, { useEffect, useState } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001"

const CATEGORY_MAP = {
  painting_json: "TL_01. 2D_02.회화(Json)",
  craft_json: "TL_01. 2D_04.공예(Json)",
  sculpture_json: "TL_01. 2D_06.조각(Json)",
}

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

  const itemsPerPage = 84
  const navigate = useNavigate()

  const realFolder = CATEGORY_MAP[category] || category

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
        const start = (page - 1) * itemsPerPage
        const currentFiles = allFiles.slice(start, start + itemsPerPage)

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

  const totalPages = Math.ceil(allFiles.length / itemsPerPage) || 1

  // 카드 클릭 → 이 화면 내에서 썸네일 목록 펼치기
  const handleCardClick = (item) => {
    setExpandedId((prev) => (prev === item.id ? null : item.id))
  }

  // 상세페이지로 넘기는 버튼 용도 (대표 id 사용)
  const handleOpenDetail = (item) => {
    const firstId = item.variants?.[0]?.id || item.id

    // ✅ 이 작품에 묶여 있는 모든 id들을 쿼리로 함께 넘김
    const variantIds = (item.variants || [{ id: item.id }])
      .map((v) => v.id)
      .filter(Boolean)
      .join(",")

    navigate(
      `/detail/${encodeURIComponent(firstId)}?category=${category}` +
        (variantIds
          ? `&variants=${encodeURIComponent(variantIds)}`
          : "")
    )
  }

  return (
    <div className="p-10 bg-white min-h-screen">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-blue-600 mb-6 text-center">
          🎨 {category.replace("_json", "").toUpperCase()} GALLERY
        </h1>

        {/* 🔻 상단 카테고리 버튼 영역 제거됨 */}

        {/* 에러 */}
        {error && (
          <div className="text-center text-red-600 mb-4 text-sm">
            {error}
          </div>
        )}

        {/* 로딩 / 카드 그리드 */}
        {loading ? (
          <div className="text-center text-gray-500 mt-20 animate-pulse">
            📡 작품 데이터를 불러오는 중입니다...
          </div>
        ) : (
          <>
            {/* 카드 그리드 */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                gap: "24px",
              }}
            >
              {pageItems.map((item, idx) => (
                <div
                  key={item.id ?? idx}
                  style={{
                    borderRadius: 18,
                    border: "1px solid rgba(0,0,0,0.06)",
                    boxShadow: "0 6px 18px rgba(15,23,42,0.12)",
                    padding: 12,
                    backgroundColor: "#ffffff",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    transition:
                      "transform 0.15s ease, box-shadow 0.15s ease",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-3px)"
                    e.currentTarget.style.boxShadow =
                      "0 10px 24px rgba(15,23,42,0.18)"
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)"
                    e.currentTarget.style.boxShadow =
                      "0 6px 18px rgba(15,23,42,0.12)"
                  }}
                >
                  {/* 대표 이미지 + 제목/작가 영역 (클릭 시 썸네일 펼치기) */}
                  <div
                    onClick={() => handleCardClick(item)}
                    style={{ width: "100%", textAlign: "center" }}
                  >
                    <img
                      src={item.img || item.variants?.[0]?.img}
                      alt={item.meta.title}
                      style={{
                        width: 160,
                        height: 160,
                        objectFit: "cover",
                        borderRadius: 18,
                        marginBottom: 10,
                      }}
                    />

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

                  {/* 상세 보기 버튼 */}
                  <button
                    type="button"
                    onClick={() => handleOpenDetail(item)}
                    style={{
                      marginTop: 10,
                      padding: "6px 12px",
                      borderRadius: 999,
                      border: "1px solid #e5e7eb",
                      backgroundColor: "#f9fafb",
                      fontSize: 12,
                      color: "#4b5563",
                      cursor: "pointer",
                    }}
                  >
                    상세 보기
                  </button>
                </div>
              ))}
            </div>

            {/* 페이지네이션 */}
            <div className="flex justify-center items-center mt-8 gap-4">
              <button
                onClick={() => setPage((p) => Math.max(p - 1, 1))}
                className="px-4 py-2 bg-gray-200 rounded-xl hover:bg-gray-300"
              >
                ◀ 이전
              </button>
              <span className="text-gray-600">
                {page} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
                className="px-4 py-2 bg-gray-200 rounded-xl hover:bg-gray-300"
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
