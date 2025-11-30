import os
import base64
import asyncio
import random
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from google import genai

# ───────────────────────────────
# 1️⃣ 환경 설정
# ───────────────────────────────
load_dotenv(Path(__file__).with_name(".env"))

API_KEY = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

JSON_BASE = os.getenv("JSON_BASE", "http://localhost:8080/json")
FIND_IMAGE_API = os.getenv("FIND_IMAGE_API", "http://localhost:8080/find_image")

client = genai.Client(api_key=API_KEY) if API_KEY else None
aclient: Optional[httpx.AsyncClient] = None


# ───────────────────────────────
# 2️⃣ FastAPI 앱 정의
# ───────────────────────────────
app = FastAPI(
    title="AI Curator Backend (With TTS Narration)",
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────────
# 3️⃣ HTTPX 클라이언트 초기화
# ───────────────────────────────
@app.on_event("startup")
async def on_startup():
    global aclient
    timeout = httpx.Timeout(20.0, connect=5.0, read=15.0)
    aclient = httpx.AsyncClient(timeout=timeout)
    print("HTTPX client started.")


@app.on_event("shutdown")
async def on_shutdown():
    global aclient
    if aclient:
        await aclient.aclose()
        print("HTTPX client closed.")


# ───────────────────────────────
# 4️⃣ 데이터 모델
# ───────────────────────────────
class CurateIn(BaseModel):
    id: str
    card: Optional[Dict[str, Any]] = None


class CompareIn(BaseModel):
    ids: List[str]
    locale: Optional[str] = "ko"
    embed_images: Optional[bool] = False


class CompareOut(BaseModel):
    analysis: str
    left: Dict[str, Any]
    right: Dict[str, Any]


class AgentIn(BaseModel):
    query: str


# ───────────────────────────────
# 5️⃣ 유틸 함수
# ───────────────────────────────
def safe_get(d: Optional[Dict], *keys, default: str = "") -> str:
    cur = d or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return str(cur) if cur is not None else default


def build_prompt(detail: Dict[str, Any]) -> str:
    """AI 설명문 프롬프트"""
    title = safe_get(detail, "Description", "ArtTitle_kor")
    artist = safe_get(detail, "Description", "ArtistName_kor")
    material = safe_get(detail, "Description", "Material_kor")
    main_cat = safe_get(detail, "Object_Info", "MainCategory")
    mid_cat = safe_get(detail, "Object_Info", "MiddleCategory")
    sub_cat = safe_get(detail, "Object_Info", "SubCategory")
    photo_date = safe_get(detail, "Photo_Info", "PhotoDate")

    return "\n".join(
        [
            "당신은 국공립 미술관의 전문 큐레이터입니다.",
            "과장 없이 정확하고 따뜻하게 설명하세요.",
            "",
            f"작품 제목: {title}",
            f"작가: {artist}",
            f"분류: {main_cat}/{mid_cat}/{sub_cat}",
            f"재질: {material}",
            f"촬영 일자: {photo_date}",
            "",
            "출력 형식:",
            "1) 작품 개요",
            "2) 시대·양식 맥락",
            "3) 형식·재료 분석",
            "4) 감상 포인트(3개)",
        ]
    )


def build_immersive_prompt(detail: Dict[str, Any]) -> str:
    """몰입형 내레이션 프롬프트"""
    title = safe_get(detail, "Description", "ArtTitle_kor", default="제목 없음")
    artist = safe_get(detail, "Description", "ArtistName_kor", default="작가 미상")
    material = safe_get(detail, "Description", "Material_kor", default="")
    main_cat = safe_get(detail, "Object_Info", "MainCategory", default="")
    mid_cat = safe_get(detail, "Object_Info", "MiddleCategory", default="")
    location = safe_get(detail, "Description", "Location_kor", default="")

    return f"""
당신은 전문 도슨트입니다.
관람객이 작품 속으로 걸어 들어가는 듯한 5단계 내레이션을 만들어 주세요.

조건:
- 총 5개의 짧은 문단
- 시선 이동 / 공간감 / 감정 포인트 포함
- 추측 금지, 사실 기반

작품 정보:
제목: {title}
작가: {artist}
분류: {main_cat}/{mid_cat}
재질: {material}
소재지: {location}

출력 형식:
(번호 없이 5개의 문단을 줄바꿈으로 나열)
    """.strip()


async def _retry_get(url: str, expect_json: bool = False):
    """Node 서버용 GET + 재시도"""
    delays = [0.0, 0.5, 1.0]
    last_exc = None

    for d in delays:
        if d:
            await asyncio.sleep(d)
        try:
            r = await aclient.get(url)
            if r.status_code == 200:
                return r.json() if expect_json else r
        except Exception as e:
            last_exc = e

    if last_exc:
        raise last_exc
    return None


async def _fetch_json(id_: str) -> Optional[Dict[str, Any]]:
    url = f"{JSON_BASE}/{id_}"
    try:
        return await _retry_get(url, expect_json=True)
    except Exception:
        return None


async def _find_image_url(id_: str) -> Optional[str]:
    url = f"{FIND_IMAGE_API}/{id_}"
    try:
        data = await _retry_get(url, expect_json=True)
        if data and isinstance(data, dict) and "url" in data:
            return f"http://localhost:8080{data['url']}"
    except Exception:
        return None
    return None


async def _fetch_image_b64(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        r = await _retry_get(url)
        if r and r.status_code == 200:
            return base64.b64encode(r.content).decode("utf-8")
    except Exception:
        return None
    return None


# ───────────────────────────────
# 6️⃣ 헬스 체크
# ───────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


# ───────────────────────────────
# 7️⃣ 단일 작품 큐레이션
# ───────────────────────────────
@app.post("/curate")
async def curate(req: CurateIn):
    if not API_KEY:
        raise HTTPException(500, "API_KEY 미설정")

    prompt = build_prompt(req.card or {})

    def _run():
        return client.models.generate_content(model=MODEL_NAME, contents=prompt)

    resp = await asyncio.to_thread(_run)
    text = getattr(resp, "text", "").strip()

    return {"curator_text": text}


# ───────────────────────────────
# 8️⃣ 두 작품 비교
# ───────────────────────────────
def _extract_meta(detail: Dict[str, Any]) -> Dict[str, Any]:
    desc = detail.get("Description", {})
    artist = desc.get("ArtistName_kor") or desc.get("ArtistName_eng") or "작가 미상"
    title = desc.get("ArtTitle_kor") or desc.get("ArtTitle_eng") or "제목 없음"
    material = desc.get("Material_kor") or ""
    year = desc.get("YearOfWork") or ""
    return {"artist": artist, "title": title, "year": year, "material": material}


@app.post("/ai/analyze-compare", response_model=CompareOut)
async def analyze_compare(req: CompareIn):
    if len(req.ids) != 2:
        raise HTTPException(400, "ids는 2개 필요")

    left_id, right_id = req.ids

    j_left, j_right, url_left, url_right = await asyncio.gather(
        _fetch_json(left_id),
        _fetch_json(right_id),
        _find_image_url(left_id),
        _find_image_url(right_id),
    )

    if not j_left or not j_right:
        raise HTTPException(404, "JSON 불러오기 실패")

    meta_left = _extract_meta(j_left) | {"id": left_id, "image_url": url_left}
    meta_right = _extract_meta(j_right) | {"id": right_id, "image_url": url_right}

    parts = [{"text": f"두 작품 비교:\n{meta_left}\n\n{meta_right}"}]

    if req.embed_images:
        b_left, b_right = await asyncio.gather(
            _fetch_image_b64(url_left),
            _fetch_image_b64(url_right),
        )
        if b_left:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b_left}})
        if b_right:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b_right}})

    def _run():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=[{"role": "user", "parts": parts}],
        )

    resp = await asyncio.to_thread(_run)
    text = getattr(resp, "text", "").strip()

    return CompareOut(
        analysis=text,
        left=meta_left,
        right=meta_right,
    )


# ───────────────────────────────
# 9️⃣ 몰입형 내레이션
# ───────────────────────────────
@app.post("/immersive")
async def immersive(req: CurateIn):
    if not API_KEY:
        raise HTTPException(500, "API_KEY 미설정")

    prompt = build_immersive_prompt(req.card or {})

    def _run():
        return client.models.generate_content(model=MODEL_NAME, contents=prompt)

    resp = await asyncio.to_thread(_run)
    raw = getattr(resp, "text", "").strip()

    steps = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]

    return {"raw_text": raw, "steps": steps}


# ───────────────────────────────
# 🔟 생성형 에이전트: 모드 자동 선택
# ───────────────────────────────
@app.post("/agent")
async def agent(req: AgentIn):
    """
    사용자 자연어 질의 → (curation | compare | inside) 중 하나 선택 후
    바로 결과(이미지 + 텍스트)를 반환
    """
    if not API_KEY:
        raise HTTPException(500, "API_KEY 미설정")

    user_query = req.query.strip() or "오늘 볼 만한 작품을 추천해줘"

    # 1) 어떤 모드가 좋을지 선택
    selector_prompt = f"""
당신은 국공립 미술관의 AI 큐레이터입니다.
사용자 요청을 보고 아래 셋 중 가장 적절한 모드 하나만 선택하세요.

- "curation" : 오늘 볼 만한 작품 하나를 골라 자세히 설명
- "compare"  : 서로 대비되는 두 작품을 골라 비교 설명
- "inside"   : 작품 속으로 들어가는 몰입형 내레이션

아무 부가 설명 없이 위 키워드 하나만 출력하세요.

사용자 요청: "{user_query}"
"""

    def _select():
        return client.models.generate_content(model=MODEL_NAME, contents=selector_prompt)

    sel_resp = await asyncio.to_thread(_select)
    decision = getattr(sel_resp, "text", "").lower()

    if "compare" in decision:
        mode = "compare"
    elif "inside" in decision or "몰입" in user_query or "속으로" in user_query:
        mode = "inside"
    else:
        mode = "curation"

    # 2) Node 서버에서 작품 목록 랜덤 가져오기
    json_list_base = JSON_BASE.rsplit("/json", 1)[0]  # "http://localhost:8080"
    lst = await _retry_get(f"{json_list_base}/json_list?limit=120", expect_json=True)
    items = (lst or {}).get("items", [])
    if not items:
        raise HTTPException(404, "작품 목록을 불러오지 못했습니다.")

    # 3) 모드에 따라 1~2개 추출
    if mode == "compare" and len(items) >= 2:
        pick = random.sample(items, 2)
    else:
        pick = [random.choice(items)]

    # 4) 비교 모드
    if mode == "compare":
        left_id = pick[0]["id"]
        right_id = pick[1]["id"]

        j_left, j_right, url_left, url_right = await asyncio.gather(
            _fetch_json(left_id),
            _fetch_json(right_id),
            _find_image_url(left_id),
            _find_image_url(right_id),
        )

        if not j_left or not j_right:
            raise HTTPException(404, "비교용 JSON 로드 실패")

        meta_left = _extract_meta(j_left) | {"id": left_id, "image_url": url_left}
        meta_right = _extract_meta(j_right) | {"id": right_id, "image_url": url_right}

        parts = [
            {
                "text": f"두 작품 비교:\n{meta_left}\n\n{meta_right}\n\n사용자 요청: {user_query}",
            }
        ]

        def _run_cmp():
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=[{"role": "user", "parts": parts}],
            )

        resp = await asyncio.to_thread(_run_cmp)
        text = getattr(resp, "text", "").strip()

        return {
            "mode": "compare",
            "query": user_query,
            "left": meta_left,
            "right": meta_right,
            "analysis": text,
        }

    # 5) 단일 작품 (curation / inside)
    art_id = pick[0]["id"]
    detail = await _fetch_json(art_id)
    if not detail:
        raise HTTPException(404, "작품 JSON 로드 실패")
    img_url = await _find_image_url(art_id)

    base_meta = _extract_meta(detail) | {"id": art_id, "image_url": img_url}

    if mode == "inside":
        prompt = build_immersive_prompt(detail)

        def _run_inside():
            return client.models.generate_content(model=MODEL_NAME, contents=prompt)

        resp = await asyncio.to_thread(_run_inside)
        raw = getattr(resp, "text", "").strip()
        steps = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]

        return {
            "mode": "inside",
            "query": user_query,
            "artwork": base_meta,
            "raw_text": raw,
            "steps": steps,
        }

    # 6) 기본: curation
    prompt = build_prompt(detail) + f"\n\n사용자 요청: {user_query}"

    def _run_cur():
        return client.models.generate_content(model=MODEL_NAME, contents=prompt)

    resp = await asyncio.to_thread(_run_cur)
    text = getattr(resp, "text", "").strip()

    return {
      "mode": "curation",
      "query": user_query,
      "artwork": base_meta,
      "curator_text": text,
    }


import re

def curate_artwork(user_query: str):
    """
    사용자의 요청을 분석하여 가장 관련성 높은 작품을 반환.
    (색상, 분위기, 태그 기반 간단한 필터링)
    """

    q = user_query.lower()

    # 색상 키워드 사전
    color_map = {
        "푸른": "blue", "파란": "blue", "파랑": "blue", "blue": "blue",
        "초록": "green", "녹색": "green", "green": "green",
        "빨강": "red", "붉은": "red", "red": "red",
        "노랑": "yellow", "노란": "yellow", "yellow": "yellow",
        "하얀": "white", "흰색": "white", "white": "white",
    }

    detected_colors = []
    for kr, eng in color_map.items():
        if kr in q:
            detected_colors.append(eng)

    # 분위기 키워드
    mood_keywords = ["calm", "peaceful", "bright", "dark", "mysterious"]
    detected_mood = [m for m in mood_keywords if m in q]

    candidates = []

    for item in ARTWORKS:  # 너의 JSON 리스트
        score = 0

        # 색상 매칭 점수
        if "colors" in item:
            for c in detected_colors:
                if c in item["colors"]:
                    score += 5

        # 태그(풍경, 인물 등)
        if "tags" in item:
            for word in q.split():
                if word in item["tags"]:
                    score += 3

        # 작품 설명 검색
        desc = item.get("description", "").lower()
        if any(k in desc for k in detected_colors):
            score += 2

        if score > 0:
            candidates.append((score, item))

    # 필터에 걸리는 작품이 없으면 랜덤 fallback
    if not candidates:
        import random
        return random.choice(ARTWORKS)

    # 최고 점수 작품 반환
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]
