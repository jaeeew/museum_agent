import React, { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"

export default function Gallery() {
  const [category, setCategory] = useState("painting_json")
  const [allFiles, setAllFiles] = useState([])
  const [page, setPage] = useState(1)
  const [pageItems, setPageItems] = useState([])
  const [loading, setLoading] = useState(true)
  const itemsPerPage = 20

  // 🆕 비교 모드 & 선택 상태
  const [compareMode, setCompareMode] = useState(false)
  const [selected, setSelected] = useState([]) // [id, id]

  const jsonBase = "http://localhost:8080/json_extracted"
  const apiBase = "http://localhost:8080/json_list"
  const findImageAPI = "http://localhost:8080/find_image"
  const navigate = useNavigate()

  // ✅ 1️⃣ 카테고리별 JSON 목록 가져오기
  useEffect(() => {
    const loadList = async () => {
      try {
        setLoading(true)
        const res = await fetch(`${apiBase}/${category}`)
        const list = await res.json()
        setAllFiles(list)
        setPage(1)
      } catch (err) {
        console.error("❌ 목록 로드 실패:", err)
      } finally {
        setLoading(false)
      }
    }
    loadList()
  }, [category])

  // ✅ 2️⃣ 현재 페이지의 JSON 데이터 로드
  useEffect(() => {
    const loadPage = async () => {
      if (!allFiles.length) return
      setLoading(true)
      try {
        const start = (page - 1) * itemsPerPage
        const currentFiles = allFiles.slice(start, start + itemsPerPage)

        const data = await Promise.all(
          currentFiles.map(async (file) => {
            const res = await fetch(`${jsonBase}/${category}/${file}`)
            if (!res.ok) throw new Error("JSON 로드 실패: " + file)
            const json = await res.json()

            // ✅ 작품명, 작가명 자동 추출 (Description 내부 포함)
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

            // ✅ 이미지 탐색 (서버 API 이용)
            const prefix = file.replace(/\.[^/.]+$/, "")
            let imgUrl = null
            try {
              const resImg = await fetch(`${findImageAPI}/${prefix}`)
              if (resImg.ok) {
                const { url } = await resImg.json()
                imgUrl = `http://localhost:8080${url}`
              }
            } catch {}

            return {
              id: prefix,
              img: imgUrl,
              meta: { title, artist, category },
            }
          })
        )

        setPageItems(data)
      } catch (err) {
        console.error("❌ 페이지 로드 실패:", err)
      } finally {
        setLoading(false)
      }
    }
    loadPage()
  }, [allFiles, page, category])

  const totalPages = Math.ceil(allFiles.length / itemsPerPage)

  // 🆕 카드 클릭 동작 (비교 모드에 따라 분기)
  const handleCardClick = (item) => {
    if (!compareMode) {
      navigate(`/detail/${encodeURIComponent(item.id)}?category=${category}`)
      return
    }
    setSelected((prev) => {
      const exists = prev.includes(item.id)
      if (exists) return prev.filter((id) => id !== item.id)
      if (prev.length >= 2) return prev // 최대 2개
      return [...prev, item.id]
    })
  }

  // 🆕 비교하기 실행
  const handleCompare = () => {
    if (selected.length !== 2) return
    const [a, b] = selected
    navigate(`/compare?ids=${encodeURIComponent(a)},${encodeURIComponent(b)}&category=${category}`)
  }

  // 🆕 비교 모드 토글 시 선택 초기화
  const toggleCompareMode = () => {
    setCompareMode((v) => !v)
    setSelected([])
  }

  return (
    <div className="p-10 bg-white min-h-screen">
      <h1 className="text-3xl font-bold text-blue-600 mb-6 text-center">
        🎨 {category.replace("_json", "").toUpperCase()} GALLERY
      </h1>

      {/* 카테고리 + 비교 컨트롤 */}
      <div className="flex flex-col gap-3 items-center mb-6">
        <div className="flex justify-center gap-4">
          {["craft_json", "painting_json", "sculpture_json"].map((cat) => (
            <button
              key={cat}
              onClick={() => {
                setCategory(cat)
                setSelected([])
                setCompareMode(false)
              }}
              className={`px-4 py-2 rounded-2xl transition ${
                category === cat
                  ? "bg-blue-500 text-white"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
            >
              {cat.replace("_json", "").toUpperCase()}
            </button>
          ))}

          {/* 🆕 두 작품 비교하기 버튼 */}
          <button
            onClick={toggleCompareMode}
            className={`px-4 py-2 rounded-2xl transition border ${
              compareMode
                ? "bg-purple-600 text-white border-purple-600"
                : "bg-white text-purple-600 border-purple-400 hover:bg-purple-50"
            }`}
          >
            🆚 두 작품 비교하기
          </button>
        </div>

        {/* 🆕 비교 모드 상태바 */}
        {compareMode && (
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-600">
              선택 {selected.length} / 2
            </span>
            <button
              onClick={handleCompare}
              disabled={selected.length !== 2}
              className={`px-4 py-2 rounded-xl transition ${
                selected.length === 2
                  ? "bg-green-600 text-white hover:bg-green-700"
                  : "bg-gray-200 text-gray-500 cursor-not-allowed"
              }`}
            >
              비교하기
            </button>
            <button
              onClick={() => setSelected([])}
              className="px-3 py-2 text-sm bg-gray-100 rounded-xl hover:bg-gray-200"
            >
              선택 초기화
            </button>
          </div>
        )}
      </div>

      {/* 로딩 상태 */}
      {loading ? (
        <div className="text-center text-gray-500 mt-20 animate-pulse">
          📡 데이터를 불러오는 중입니다...
        </div>
      ) : (
        <>
          {/* 카드뷰 */}
          <div className="grid grid-cols-5 gap-6">
            {pageItems.map((item, idx) => {
              const isSelected = selected.includes(item.id)
              return (
                <div
                  key={idx}
                  onClick={() => handleCardClick(item)}
                  className={`relative border rounded-2xl shadow transition p-3 flex flex-col items-center cursor-pointer ${
                    compareMode
                      ? isSelected
                        ? "ring-2 ring-purple-500"
                        : "hover:shadow-lg"
                      : "hover:shadow-lg"
                  }`}
                >
                  {/* 🆕 체크박스 오버레이 */}
                  {compareMode && (
                    <div className="absolute top-2 right-2">
                      <input
                        type="checkbox"
                        readOnly
                        checked={isSelected}
                        className="w-5 h-5 accent-purple-600"
                      />
                    </div>
                  )}

                  {item.img ? (
                    <img
                      src={item.img}
                      alt={item.meta.title}
                      className="w-40 h-40 object-cover rounded-xl mb-3"
                    />
                  ) : (
                    <div className="w-40 h-40 flex items-center justify-center bg-gray-100 rounded-xl mb-3 text-gray-400 text-sm">
                      이미지 없음
                    </div>
                  )}
                  {/* 🔹 작품 이름 */}
                  <p className="text-sm font-semibold text-gray-700 text-center line-clamp-2">
                    {item.meta.title}
                  </p>
                  {/* 🔹 작가 이름 */}
                  <p className="text-xs text-gray-500">{item.meta.artist}</p>
                </div>
              )
            })}
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
              {page} / {totalPages || 1}
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
  )
}
