import os
import base64
import asyncio
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, HTTPException, Request
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


JSON_BASE = os.getenv("JSON_BASE", "http://localhost:8080/json_extracted")
FIND_IMAGE_API = os.getenv("FIND_IMAGE_API", "http://localhost:8080/find_image")

client = genai.Client(api_key=API_KEY) if API_KEY else None
aclient: Optional[httpx.AsyncClient] = None


# ───────────────────────────────
# 2️⃣ FastAPI 앱 정의
# ───────────────────────────────
app = FastAPI(
    title="AI Curator Backend (Optimized)",
    description="Gemini API를 이용한 작품 설명 및 비교 분석 서버",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────
# 3️⃣ 미들웨어: 요청 시간 측정
# ───────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    import time
    start = time.perf_counter()
    response = await call_next(request)
    dur = (time.perf_counter() - start) * 1000
    print(f"[TIMING] {request.method} {request.url.path} -> {dur:.1f} ms")
    return response


# ───────────────────────────────
# 4️⃣ 모델 / HTTPX 클라이언트 초기화
# ───────────────────────────────
@app.on_event("startup")
async def on_startup():
    global aclient
    # 기존: httpx.Timeout(5.0, connect=3.0, read=5.0)
    timeout = httpx.Timeout(15.0, connect=5.0, read=15.0)
    aclient = httpx.AsyncClient(timeout=timeout)
    print("HTTPX client started.")

@app.on_event("shutdown")
async def on_shutdown():
    global aclient
    if aclient:
        await aclient.aclose()
        print("HTTPX client closed.")


# ───────────────────────────────
# 5️⃣ 스키마
# ───────────────────────────────
class CurateIn(BaseModel):
    id: str
    card: Optional[Dict[str, Any]] = None


class CompareIn(BaseModel):
    ids: List[str]
    category: str
    locale: Optional[str] = "ko"
    embed_images: Optional[bool] = False


class CompareOut(BaseModel):
    analysis: str
    left: Dict[str, Any]
    right: Dict[str, Any]


# ───────────────────────────────
# 6️⃣ 유틸리티 함수
# ───────────────────────────────
def safe_get(d: Optional[Dict], *keys, default: str = "") -> str:
    cur = d or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return str(cur) if cur is not None else default


def build_prompt(detail: Dict[str, Any]) -> str:
    title = safe_get(detail, "Description", "ArtTitle_kor")
    artist = safe_get(detail, "Description", "ArtistName_kor")
    material = safe_get(detail, "Description", "Material_kor")
    main_cat = safe_get(detail, "Object_Info", "MainCategory")
    mid_cat = safe_get(detail, "Object_Info", "MiddleCategory")
    sub_cat = safe_get(detail, "Object_Info", "SubCategory")
    photo_date = safe_get(detail, "Photo_Info", "PhotoDate")

    return "\n".join([
        "당신은 국공립 미술관의 전문 큐레이터입니다.",
        "과장 없이 정확하고 품위 있게 한국어로 설명하세요.",
        "",
        f"작품 제목: {title}",
        f"작가: {artist}",
        f"분류: {main_cat} / {mid_cat} / {sub_cat}",
        f"재질: {material}",
        f"촬영 일자: {photo_date}",
        "",
        "출력 형식:",
        "1) 작품 개요",
        "2) 시대·양식적 맥락",
        "3) 형식·재료 분석",
        "4) 감상 포인트 — 3개 불릿",
        "5) 참고 — 저작권 및 데이터 이용범위",
    ])


def _extract_meta(detail: Dict[str, Any]) -> Dict[str, Any]:
    desc = (detail or {}).get("Description", {}) or {}
    title = desc.get("ArtTitle_kor") or desc.get("ArtTitle_eng") or "제목 없음"
    artist = desc.get("ArtistName_kor") or desc.get("ArtistName_eng") or "작가 미상"
    year = desc.get("YearOfWork") or ""
    material = desc.get("Material_kor") or desc.get("Material") or ""
    size = desc.get("Size") or ""
    return {"title": title, "artist": artist, "year": year, "material": material, "size": size}


def _build_compare_prompt(meta_left, meta_right, json_left, json_right, locale="ko") -> str:
    guide = """
 당신은 국공립 미술관의 전문 큐레이터입니다. 차분하고 따뜻한 말투로, 관람객에게 편안히 이야기하듯 한국어 구어체로 두 작품을 비교해 설명하세요.
설명은 2~3개의 짧은 단락, 총 6~8문장으로 작성합니다. 제목, 번호, 불릿, 이모지, 괄호 표시는 사용하지 마세요.
첫 단락에서 두 작품의 핵심 의미나 주제를 1~2문장으로 선명하게 제시하고, 공통점과 큰 차이를 한눈에 grasp할 수 있게 간결히 말해 주세요.
다음 단락에서는 시대와 지역, 재료와 기법, 화면 구성이나 선의 리듬, 색의 대비, 주제 모티프 같은 요소를 꼭 필요한 2~3포인트로만 짧게 비교하세요. 수치나 연대, 치수처럼 확인 가능한 사실은 부드럽게 자연어로 녹여 전하고, 근거는 카드나 검색 컨텍스트에 포함된 정보만 사용하세요.
컨텍스트에 없는 정보는 추정하지 않습니다. 불확실한 내용은 단정하지 말고 ~로 보입니다, ~로 추정됩니다, 확인되지 않았습니다처럼 신중히 표현하세요. 작가명과 제작 연도 등 시기적 불일치 의심이 있으면 한 문장으로 조심스럽게 짚되 감상 흐름을 해치지 않도록 간단히 처리하세요. 권리나 이용 범위, 라이선스, 파일 경로와 같은 기술적 표기는 언급하지 마세요.
마지막 문장은 관람 팁으로 마무리하세요. 두 작품을 어디에서부터 보면 좋은지, 어떤 디테일을 나란히 보면 차이가 또렷해지는지 한두 문장으로 권유형 어조로 안내하세요.
    """.strip()
    lang = "한국어" if locale.startswith("ko") else "English"
    return f"""
응답 언어: {lang}

[작품 A]
{meta_left}

[작품 B]
{meta_right}

[JSON 일부 A]
{str(json_left)[:4000]}

[JSON 일부 B]
{str(json_right)[:4000]}

{guide}
    """.strip()


# ───────────────────────────────
# 7️⃣ 외부 fetch 헬퍼
# ───────────────────────────────

# ── 재시도 유틸 (지수 백오프) ──
async def _retry_get(url: str, expect_json: bool = False):
    delays = [0.0, 0.7, 1.5, 3.0]  # 총 3회 재시도
    last_exc = None
    for d in delays:
        if d:
            await asyncio.sleep(d)
        try:
            r = await aclient.get(url)
            if r.status_code == 200:
                return r.json() if expect_json else r
            print("GET non-200:", url, r.status_code)
        except Exception as e:
            last_exc = e
            print("GET exception:", url, e)
    if last_exc:
        raise last_exc
    return None

async def _fetch_json(category: str, id_: str) -> Optional[Dict[str, Any]]:
    url = f"{JSON_BASE}/{category}/{id_}.json"
    try:
        return await _retry_get(url, expect_json=True)
    except Exception as e:
        print("fetch_json err", url, e)
        return None

async def _find_image_url(id_: str) -> Optional[str]:
    url = f"{FIND_IMAGE_API}/{id_}"
    try:
        data = await _retry_get(url, expect_json=True)
        if isinstance(data, dict):
            p = data.get("url")
            if p:
                return p if p.startswith("http") else f"http://localhost:8080{p}"
        return None
    except Exception as e:
        print("find_image_url err", url, e)
        return None

async def _fetch_image_b64(img_url: Optional[str]) -> Optional[str]:
    if not img_url:
        return None
    try:
        r = await _retry_get(img_url, expect_json=False)
        if r and r.status_code == 200:
            return base64.b64encode(r.content).decode("utf-8")
    except Exception as e:
        print("fetch_image err", img_url, e)
    return None

# ───────────────────────────────
# 8️⃣ 헬스 체크
# ───────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "has_api_key": bool(API_KEY)}


# ───────────────────────────────
# 9️⃣ 큐레이션 생성 API
# ───────────────────────────────
@app.post("/curate")
async def curate(req: CurateIn):
    if not API_KEY or client is None:
        raise HTTPException(500, "GOOGLE_GENAI_API_KEY가 설정되지 않았습니다.")
    prompt = build_prompt(req.card or {})

    try:
        def _call_llm():
            return client.models.generate_content(model=MODEL_NAME, contents=prompt)
        response = await asyncio.to_thread(_call_llm)
        text = getattr(response, "text", "") or (
            response.candidates[0].content.parts[0].text
            if getattr(response, "candidates", None)
            else ""
        )
        return {"curator_text": text.strip() or "설명문 생성 실패"}
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")


# ───────────────────────────────
# 🔟 비교 분석 API
# ───────────────────────────────
@app.post("/ai/analyze-compare", response_model=CompareOut)
async def analyze_compare(req: CompareIn):
    if not API_KEY or client is None:
        raise HTTPException(500, "GOOGLE_GENAI_API_KEY가 설정되지 않았습니다.")
    if len(req.ids) != 2:
        raise HTTPException(400, "ids는 정확히 2개여야 합니다.")

    left_id, right_id = req.ids

    # JSON + 이미지 URL 병렬 요청
    j_left, j_right, left_img, right_img = await asyncio.gather(
        _fetch_json(req.category, left_id),
        _fetch_json(req.category, right_id),
        _find_image_url(left_id),
        _find_image_url(right_id),
    )

    if not j_left or not j_right:
        raise HTTPException(502, "작품 JSON 로드 실패")

    meta_left = _extract_meta(j_left) | {"id": left_id, "image_url": left_img}
    meta_right = _extract_meta(j_right) | {"id": right_id, "image_url": right_img}

    prompt = _build_compare_prompt(meta_left, meta_right, j_left, j_right, req.locale or "ko")

    parts = [{"text": prompt}]
    if req.embed_images:
        left_b64, right_b64 = await asyncio.gather(
            _fetch_image_b64(left_img),
            _fetch_image_b64(right_img),
        )
        if left_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": left_b64}})
        if right_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": right_b64}})

    try:
        def _call_llm():
            return client.models.generate_content(model=MODEL_NAME, contents=[{"role": "user", "parts": parts}])
        resp = await asyncio.to_thread(_call_llm)
        text = getattr(resp, "text", "") or (
            resp.candidates[0].content.parts[0].text
            if getattr(resp, "candidates", None)
            else ""
        )
    except Exception as e:
        raise HTTPException(502, f"Gemini 호출 실패: {e}")

    return CompareOut(analysis=text.strip(), left=meta_left, right=meta_right)
