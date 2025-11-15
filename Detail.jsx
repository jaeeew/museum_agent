import React, { useEffect, useState } from "react"
import { useParams, useSearchParams, Link } from "react-router-dom"

const NODE_API = "http://localhost:8080"   // JSON/이미지용 Node 서버
const FAST_API = "http://127.0.0.1:8000"   // AI 설명문용 FastAPI 서버

export default function Detail() {
  const { id } = useParams()
  const [searchParams] = useSearchParams()
  const category = searchParams.get("category")

  const [data, setData] = useState(null)
  const [imgUrl, setImgUrl] = useState(null)
  const [curation, setCuration] = useState("")        // 🧠 AI 설명문
  const [loadingCuration, setLoadingCuration] = useState(false)
  const [showCuration, setShowCuration] = useState(false) // 펼치기/접기

  // -------------------- 데이터 로드 --------------------
  useEffect(() => {
    const loadDetail = async () => {
      try {
        const jsonUrl = `${NODE_API}/json_extracted/${category}/${id}.json`
        const res = await fetch(jsonUrl)
        const json = await res.json()
        setData(json)

        const imgRes = await fetch(`${NODE_API}/find_image/${id}`)
        if (imgRes.ok) {
          const { url } = await imgRes.json()
          setImgUrl(`${NODE_API}${url}`)
        }
      } catch (err) {
        console.error("❌ 상세정보 로드 실패:", err)
      }
    }
    loadDetail()
  }, [id, category])

  // -------------------- AI 설명문 생성 --------------------
  const handleCurateClick = async () => {
    const next = !showCuration
    setShowCuration(next)

    // 이미 받아왔으면 재요청 없이 토글만
    if (curation || !next || !data) return

    setLoadingCuration(true)
    try {
      console.log("🧠 요청 →", `${FAST_API}/curate`)
      const res = await fetch(`${FAST_API}/curate`, {
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

  // -------------------- 로딩 상태 --------------------
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center h-screen text-gray-500">
        📡 상세 정보를 불러오는 중...
      </div>
    )
  }

  // -------------------- 데이터 정리 --------------------
  const desc = data.Description || {}
  const obj = data.Object_Info || {}
  const photo = data.Photo_Info || {}
  const image = data.Image_Info || {}
  const datainfo = data.Data_Info || {}

  const titleKor = desc.ArtTitle_kor || data.title || "제목 없음"
  const artistKor = desc.ArtistName_kor || "작가 미상"
  const locationKor = desc.Location_kor || "-"
  const materialKor = desc.Material_kor || "-"
  const categoryKor = desc.Class_kor || obj.MiddleCategory || "-"

  // -------------------- 화면 렌더링 --------------------
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-10">
      <Link
        to="/gallery"
        className="mb-6 text-blue-500 hover:underline text-sm"
      >
        ← 갤러리로 돌아가기
      </Link>

      <div className="bg-white rounded-2xl shadow-lg p-8 w-[900px]">
        {/* 제목 */}
        <h1 className="text-2xl font-bold text-center mb-4 text-blue-700">
          {titleKor}
        </h1>

        {/* 작가 */}
        <p className="text-center text-gray-600 mb-6">{artistKor}</p>

        {/* 대표 이미지 */}
        {imgUrl && (
          <img
            src={imgUrl}
            alt={titleKor}
            className="w-full h-[450px] object-contain rounded-xl mb-6 shadow"
          />
        )}

        {/* 주요 정보 */}
        <div className="grid grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-100 rounded-xl p-4">
            <h3 className="font-semibold text-blue-600 mb-2">📘 기본 정보</h3>
            <p><strong>분류:</strong> {categoryKor}</p>
            <p><strong>시대:</strong> {obj.MainCategory || "정보 없음"}</p>
            <p><strong>소분류:</strong> {obj.SubCategory || "정보 없음"}</p>
            <p><strong>재질:</strong> {materialKor}</p>
            <p><strong>소재지:</strong> {locationKor}</p>
          </div>

          <div className="bg-gray-100 rounded-xl p-4">
            <h3 className="font-semibold text-blue-600 mb-2">📷 촬영 정보</h3>
            <p><strong>촬영일자:</strong> {photo.PhotoDate || "정보 없음"}</p>
            <p><strong>촬영장비:</strong> {photo.PhotoEquipment || "정보 없음"}</p>
            <p><strong>이미지 크기:</strong> 
              {image.Width ? `${image.Width} x ${image.Length} x ${image.Height || "-"}` : "정보 없음"}
            </p>
            <p><strong>파일명:</strong> {datainfo.ImageFileName || id}</p>
            <p><strong>형식:</strong> {datainfo.SourceDataExtension || "jpg"}</p>
            <p><strong>이용범위:</strong> {datainfo.Rangeofuse || "-"}</p>
          </div>
        </div>

        {/* 작품 설명(원본 메타에서 추출) */}
        <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
          <h3 className="font-semibold text-blue-600 mb-2">🖋️ 작품 설명</h3>
          <p className="text-gray-700 leading-relaxed">
            {desc.ArtTitle_kor && desc.ArtTitle_eng ? (
              <>
                <strong>{desc.ArtTitle_kor}</strong>
                <br />
                <span className="text-gray-500 italic">{desc.ArtTitle_eng}</span>
              </>
            ) : (
              "작품 설명 없음"
            )}
          </p>
        </div>

        {/* 🧠 AI 설명문 (버튼 + 아코디언) */}
        <div className="mt-6">
          <button
            onClick={handleCurateClick}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-xl shadow transition"
          >
            {showCuration ? "🧠 AI 설명문 접기" : "🧠 AI 설명문 생성"}
          </button>

          {showCuration && (
            <div className="mt-4 bg-gray-100 rounded-xl p-4 text-sm text-gray-700 border border-gray-200">
              {loadingCuration ? (
                <p className="text-blue-500 animate-pulse">⌛ 설명문 생성 중입니다...</p>
              ) : (
                <p className="whitespace-pre-wrap leading-relaxed">
                  {curation || "아직 생성된 설명문이 없습니다."}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
