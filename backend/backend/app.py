import os
import json
import random
from typing import Optional, Dict, List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google.cloud import texttospeech
import base64  # audio를 base64로 전달할 거라서
from pydantic import BaseModel

# 🆕 CLIP / torch
import torch
import open_clip  # pip install open_clip_torch

# 🆕 chromadb 타입은 아래에서 이미 쓰고 있어서 여기서 import 해도 됨
import chromadb
from chromadb import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

# ───────────────────────────────────────────────────────────
# 환경설정
# ───────────────────────────────────────────────────────────
load_dotenv()  # .env 지원

import google.generativeai as genai

API_KEY = os.environ.get("GOOGLE_API_KEY")            # 필수
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-004")

if API_KEY:
    genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ───────────────────────────────────────────────────────────
# 데이터 루트 (작품 JSON/이미지 경로)
# ───────────────────────────────────────────────────────────
DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"D:\Exhibit"))
JSON_ROOT = DATA_ROOT / "json_extracted"
IMG_ROOT  = DATA_ROOT / "image_extracted"

# ───────────────────────────────────────────────────────────
# 카테고리 별명 → 실제 폴더명 매핑
# ───────────────────────────────────────────────────────────
CATEGORY_MAP: Dict[str, str] = {
    "painting_json": "TL_01. 2D_02.회화(Json)",
    "craft_json":    "TL_01. 2D_04.공예(Json)",
    "sculpture_json": "TL_01. 2D_06.조각(Json)",
}

def map_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    return CATEGORY_MAP.get(cat, cat)

# ───────────────────────────────────────────────────────────
# Chroma(벡터DB) + Gemini 텍스트 RAG 컬렉션
#            + CLIP 이미지 임베딩 컬렉션
# ───────────────────────────────────────────────────────────
use_retriever = True          # 텍스트 RAG 사용 여부
use_image_retriever = True    # 이미지 RAG 사용 여부

retrieval = None              # curator_corpus (텍스트)
image_collection = None       # curator_image_clip (이미지)

# ✅ Gemini 텍스트 임베딩용
class GeminiEF(EmbeddingFunction):
    def __call__(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            r = genai.embed_content(model=EMBED_MODEL, content=t)
            out.append(r["embedding"])
        return out

# ✅ chroma 클라이언트 & 컬렉션 설정
try:
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )

    # 1) 텍스트 RAG용 컬렉션 (이미 build_index.py로 채워둔 것)
    retrieval = client.get_or_create_collection(
        name="curator_corpus",
        embedding_function=GeminiEF(),
        metadata={"hnsw:space": "cosine"},
    )

    # 2) 이미지 RAG용 컬렉션 (build_image_index_clip.py로 채워둔 것)
    #    → 여기에는 embedding_function 없음 (우리가 직접 CLIP 임베딩 넣고, 쿼리할 때도 직접 벡터 넣음)
    image_collection = client.get_or_create_collection(
        name="curator_image_clip",
        metadata={"hnsw:space": "cosine"},
    )

except Exception as e:
    print("[WARN] Chroma 초기화 실패:", e)
    use_retriever = False
    use_image_retriever = False
    retrieval = None
    image_collection = None

# ───────────────────────────────────────────────────────────
# CLIP 모델 (텍스트→이미지 검색용)
# ───────────────────────────────────────────────────────────
try:
    CLIP_MODEL_NAME = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    clip_model = clip_model.to(clip_device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    @torch.no_grad()
    def embed_clip_text(texts: List[str]) -> List[List[float]]:
        """
        CLIP 텍스트 인코더로 문장을 임베딩.
        → 이미지 임베딩과 같은 공간(코사인)에서 비교 가능.
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = clip_tokenizer(texts).to(clip_device)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().tolist()

except Exception as e:
    print("[WARN] CLIP 초기화 실패:", e)
    use_image_retriever = False


# ───────────────────────────────────────────────────────────
# FastAPI 앱
# ───────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # 프론트 개발 서버
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # 필요하면 추가
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지 정적 서빙
app.mount(
    "/image_extracted",
    StaticFiles(directory=str(IMG_ROOT)),
    name="images",
)

# JSON 정적 서빙
app.mount(
    "/json_extracted",
    StaticFiles(directory=str(JSON_ROOT)),
    name="json",
)

# ───────────────────────────────────────────────────────────
# 모델 입력 스키마
# ───────────────────────────────────────────────────────────
class CurateIn(BaseModel):
    id: str
    card: Optional[Dict] = None  # py3.9 호환

class CompareIn(BaseModel):
    """
    두 작품 비교용 입력.
    - ids: ["작품A_id", "작품B_id"]
    - category: "painting_json" / "craft_json" / "sculpture_json"
    - locale: 현재는 "ko"만 사용하지만 확장 대비
    """
    ids: List[str]
    category: Optional[str] = None
    locale: Optional[str] = "ko"

class AgentIn(BaseModel):
    """
    첫 화면(Welcome)에서 자연어 한 줄 입력을 받아
    - action: 어떤 화면으로 보낼지 (curate / compare / tts)
    - primary_id / secondary_id: 어떤 작품(들)을 보여줄지
    - category: 기본 카테고리 힌트
    를 결정하는 에이전트 입력 스키마
    """
    query: str
    category: Optional[str] = None
    locale: Optional[str] = "ko"

class TtsIn(BaseModel):
    """
    설명 텍스트를 받아 Google Cloud TTS로 음성을 생성하는 입력 스키마
    """
    text: str
    language_code: Optional[str] = "ko-KR"
    voice_name: Optional[str] = None   # e.g. "ko-KR-Standard-A"
    speaking_rate: Optional[float] = 1.0

# ───────────────────────────────────────────────────────────
# 유틸: 카드 로딩
# ───────────────────────────────────────────────────────────
def load_card_by_id(category: Optional[str], art_id: str) -> Dict:
    """
    json_extracted 아래에서 id와 같은 파일명을 가진 JSON을 찾아 로드.
    - category가 주어지면 json_extracted/{mapped_category}/*.json 안에서 검색
    - category가 None이면 json_extracted 전체를 순회 (느릴 수 있음)
    """
    if category:
        real_cat = map_category(category)
        candidates = [JSON_ROOT / real_cat]
    else:
        candidates = [p for p in JSON_ROOT.iterdir() if p.is_dir()]

    for cat_dir in candidates:
        if not cat_dir.is_dir():
            continue
        target = cat_dir / f"{art_id}.json"
        if target.exists():
            with target.open("r", encoding="utf-8") as f:
                card = json.load(f)
            card.setdefault("id", art_id)
            return card

    raise HTTPException(
        status_code=404,
        detail=f"card not found for id={art_id}, category={category}",
    )

# ───────────────────────────────────────────────────────────
# RAG 유틸
# ───────────────────────────────────────────────────────────
def build_query(card: Dict) -> str:
    """카드의 핵심 필드로 의미검색용 질의문을 조립"""
    parts: List[Optional[str]] = [
        card.get("title") or card.get("title_ko") or card.get("title_en"),
        card.get("artist") or card.get("artist_ko") or card.get("artist_en"),
        card.get("class") or card.get("class_ko") or card.get("class_en"),
        " ".join(card.get("categories", []) or []),
        card.get("material") or card.get("material_ko") or card.get("material_en"),
        card.get("date_or_period") or card.get("photo_date"),
    ]
    return " ".join([p for p in parts if p])

def retrieve_context(query: str, k: int = 5) -> List[Dict]:
    """Chroma에서 Top-k 검색 결과 반환. 리트리버가 비활성화면 빈 리스트."""
    if not use_retriever or not retrieval or not query:
        return []
    res = retrieval.query(query_texts=[query], n_results=k)
    if not res or not res.get("ids"):
        return []

    hits: List[Dict] = []
    ids = res["ids"][0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(ids)

    for i in range(len(ids)):
        hits.append({
            "id": ids[i],
            "text": docs[i] if i < len(docs) else "",
            "meta": metas[i] if i < len(metas) else {},
            "score": dists[i] if i < len(dists) else None,
        })
    return hits

def format_context(hits: List[Dict]) -> str:
    """모델 프롬프트에 붙일 컨텍스트 블록 문자열 생성"""
    if not hits:
        return "(관련 자료 검색 결과 없음)"
    lines: List[str] = []
    for h in hits:
        m = h.get("meta") or {}
        head = f"■ {m.get('title','(제목 미상)')} / {m.get('artist','')} / {m.get('class','')}"
        tail = f"[재질:{m.get('material','')}, 연도:{m.get('year','')}]"
        lines.append(head)
        lines.append(h.get("text", ""))
        lines.append(tail)
        lines.append("")
    return "\n".join(lines)

def retrieve_image_context(query: str, k: int = 5) -> List[Dict]:
    """
    CLIP 텍스트 임베딩으로 curator_image_clip에서
    '이미지 기반' Top-k 작품을 찾음.
    반환 형식은 텍스트 RAG와 비슷하게 맞춰서 사용하기 쉽게.
    """
    if not use_image_retriever or not image_collection or not query:
        return []

    try:
        vec = embed_clip_text([query])[0]  # 1개 쿼리 → 1벡터
    except Exception as e:
        print("[WARN] CLIP embed 실패:", e)
        return []

    res = image_collection.query(
        query_embeddings=[vec],
        n_results=k,
    )
    if not res or not res.get("ids"):
        return []

    hits: List[Dict] = []
    ids = res["ids"][0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(ids)

    for i in range(len(ids)):
        hits.append(
            {
                "id": ids[i],
                "meta": metas[i] if i < len(metas) else {},
                "score": dists[i] if i < len(dists) else None,
            }
        )
    return hits


# ───────────────────────────────────────────────────────────
# 프롬프트 빌더
# ───────────────────────────────────────────────────────────
def build_prompt(card: Dict, context_block: str) -> str:
    card = card or {}
    desc = card.get("Description") or {}
    photo = card.get("Photo_Info") or {}
    data_info = card.get("Data_Info") or {}

    # ✅ Detail.jsx 에서 쓰는 로직과 최대한 맞춤
    title = (
        card.get("title")
        or desc.get("ArtTitle_kor")
        or desc.get("ArtTitle_eng")
        or data_info.get("ImageFileName")
        or card.get("id", "")
    )

    artist = (
        card.get("artist")
        or desc.get("ArtistName_kor")
        or desc.get("ArtistName_eng")
        or ""
    )

    klass = (
        card.get("class")
        or desc.get("Class_kor")
        or desc.get("Class_eng")
        or ""
    )

    material = (
        card.get("material")
        or desc.get("Material_kor")
        or desc.get("Material_eng")
        or ""
    )

    year = (
        card.get("date_or_period")
        or photo.get("PhotoDate")
        or ""
    )

    cats = " / ".join(card.get("categories", []) or [])
    lines = [
        "당신은 국공립 미술관의 전문 큐레이터입니다. 차분하고 따뜻한 말투로, 관람객에게 편안히 이야기하듯 한국어 구어체로 설명하세요.",
        "설명은 3~4개의 짧은 단락, 총 5~7문장으로 작성합니다. 제목/번호/불릿/이모지/괄호 표시는 사용하지 마세요.",
        "이 작품이 말하는 ‘핵심 의미/주제’를 서두 1~2문장에서 선명하게 제시하고, 나머지 정보는 배경 수준으로만 간결히 덧붙이세요.",
        "형식·재료 분석은 핵심 1~2포인트(구도/필획의 강약·리듬/농담 대비 등)만 짧게 언급하세요. 감상 포인트도 1~2문장으로 권유형 종결을 사용하세요.",
        "권리·이용범위·라이선스·파일 경로(json_path 등)·출처 표기는 언급하지 마세요.",
        "카드/검색 컨텍스트에 없는 정보는 추정하지 말고, 불확실하면 ‘~로 보입니다/추정됩니다/확인되지 않았습니다’처럼 신중히 표현하세요.",
        "작가명과 제작 연도 간 시기 불일치가 의심되면, 단정하지 않고 한 문장으로 조심스럽게 짚되 주제 감상에 방해되지 않도록 간단히 처리하세요.",
        "",
        "### [검색 컨텍스트]",
        context_block,
        "",
        "### [요청 카드]",
        f"작품 제목: {title}",
        f"작가: {artist}",
        f"분류/장르: {klass}",
        f"카테고리: {cats}",
        f"재질: {material}",
        f"연도/시기: {year}",
        "",
        "최종 출력에는 위의 메타/지침 섹션을 포함하지 말고, 단락 텍스트만 제시하세요.",
    ]
    return "\n".join(lines)

# ───────────────────────────────────────────────────────────
# 랜덤 작품 선택 유틸 (Agent fallback 용)
# ───────────────────────────────────────────────────────────
def list_ids_for_category(category: str = "painting_json") -> List[str]:
    """
    주어진 카테고리에서 json 파일들의 prefix(id) 목록을 반환.
    ex) TL_01. 2D_02.회화(Json)/kart_2d000123-...json -> "kart_2d000123-..."
    """
    real_cat = map_category(category)
    target_dir = JSON_ROOT / real_cat
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    ids: List[str] = []
    for p in target_dir.glob("*.json"):
        ids.append(p.stem)  # 확장자 제거한 파일명
    return ids

def pick_random_id(category: str = "painting_json") -> Optional[str]:
    """
    해당 카테고리에서 임의의 작품 id 하나를 반환.
    작품이 없으면 None.
    """
    ids = list_ids_for_category(category)
    if not ids:
        return None
    return random.choice(ids)

# ───────────────────────────────────────────────────────────
# 라우트
# ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "embed_model": EMBED_MODEL,
        "has_api_key": bool(API_KEY),
        "retriever_enabled": bool(use_retriever),
    }

@app.get("/json_list/{category}")
def json_list(category: str):
    """
    프론트에서 쓰는 별명(category)을 받아
    실제 폴더명으로 매핑한 뒤 JSON 파일 목록을 반환.
    ex) GET /json_list/painting_json -> ["kart_2d000496-C-8-81-1.json", ...]
    """
    real_cat = map_category(category)
    target_dir = JSON_ROOT / real_cat

    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"category not found: {real_cat}")

    files = [p.name for p in target_dir.glob("*.json")]
    return files

@app.get("/find_image/{prefix}")
def find_image(prefix: str):
    """
    image_extracted 아래에서 {prefix}로 시작하는 이미지 파일을 찾고,
    가장 먼저 찾은 1개의 URL을 반환.
    """
    exts = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    for ext in exts:
        pattern = f"{prefix}*.{ext}"
        matches = list(IMG_ROOT.rglob(pattern))
        if matches:
            rel = matches[0].relative_to(IMG_ROOT)
            url = f"/image_extracted/{rel.as_posix()}"
            return {"url": url}

    raise HTTPException(status_code=404, detail=f"image not found for prefix={prefix}")

@app.post("/curate")
async def curate(req: CurateIn):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")

    card = req.card or {}

    query = build_query(card)
    hits = retrieve_context(query, k=5)
    context_block = format_context(hits)
    prompt = build_prompt(card, context_block)

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return {
        "curator_text": text,
        "retrieved": [
            {"meta": h.get("meta"), "score": h.get("score"), "id": h.get("id")}
            for h in hits
        ],
    }

@app.get("/search")
def search(q: str, k: int = 5):
    """의미검색만 단독 제공하고 싶을 때 사용."""
    if not use_retriever:
        return {"results": [], "note": "retriever disabled"}
    hits = retrieve_context(q, k=k)
    return {
        "results": [
            {
                "title": (h.get("meta") or {}).get("title"),
                "artist": (h.get("meta") or {}).get("artist"),
                "class": (h.get("meta") or {}).get("class"),
                "material": (h.get("meta") or {}).get("material"),
                "year": (h.get("meta") or {}).get("year"),
                "score": h.get("score"),
                "id": h.get("id"),
            }
            for h in hits
        ]
    }

@app.get("/search_image")
def search_image(q: str, k: int = 5):
    """
    CLIP 기반 '이미지 느낌' 검색.
    query(텍스트)를 CLIP 텍스트 임베딩으로 바꾼 뒤
    curator_image_clip 컬렉션에서 가장 비슷한 이미지들을 찾는다.
    """
    if not use_image_retriever:
        return {"results": [], "note": "image retriever disabled"}

    hits = retrieve_image_context(q, k=k)
    return {
        "results": [
            {
                "id": h.get("id"),
                "title": (h.get("meta") or {}).get("title"),
                "artist": (h.get("meta") or {}).get("artist"),
                "class": (h.get("meta") or {}).get("class"),
                "year": (h.get("meta") or {}).get("year"),
                "image_path": (h.get("meta") or {}).get("image_path"),
                "score": h.get("score"),
            }
            for h in hits
        ]
    }


@app.post("/ai/agent")
async def agent_route(req: AgentIn):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")

    # 0) 키워드 먼저 판별
    lower_q = q.lower()
    compare_keywords = ["비교", "두 작품", "두 점", "2점", "vs", "차이"]
    tts_keywords = ["읽어줘", "읽어 줘", "설명 들어보고 싶어", "음성", "tts", "음성으로","들려줘", "들려 줘"]

    is_compare = any(kw in lower_q for kw in compare_keywords)
    is_tts = any(kw in lower_q for kw in tts_keywords)

    fallback_cat = req.category or "painting_json"

        # 1) 의미 검색 (후보 작품 리스트)
    #    - 텍스트 RAG(Gemini) + 이미지 RAG(CLIP) 둘 다에서 후보를 가져와 합침
    text_hits = retrieve_context(q, k=4)          # 메타데이터 기반
    image_hits = retrieve_image_context(q, k=4)   # 시각적/스타일 기반

    candidates: List[Dict] = []
    seen_ids = set()

    # 우선 텍스트 RAG 결과
    for h in text_hits:
        m = h.get("meta") or {}
        cid = h.get("id")
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        candidates.append(
            {
                "id": cid,
                "title": m.get("title", ""),
                "artist": m.get("artist", ""),
                "class": m.get("class", ""),
                "material": m.get("material", ""),
                "year": m.get("year", ""),
            }
        )

    # 그 다음 이미지 RAG 결과 (중복 id는 건너뜀)
    for h in image_hits:
        m = h.get("meta") or {}
        cid = h.get("id")
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        candidates.append(
            {
                "id": cid,
                "title": m.get("title", ""),
                "artist": m.get("artist", ""),
                "class": m.get("class", ""),
                "material": m.get("material", ""),
                "year": m.get("year", ""),
            }
        )

    print("[agent_route] query:", q)
    print("[agent_route] candidates:", [c["id"] for c in candidates])


    # 2) '비교' 요청이면 → hits가 없어도 무조건 compare로
    if is_compare:
        selected_ids: List[str] = []

        # 검색 결과에서 먼저 채우고
        for c in candidates:
            if c["id"] and c["id"] not in selected_ids:
                selected_ids.append(c["id"])
            if len(selected_ids) >= 2:
                break

        # 부족하면 랜덤으로 채우기
        while len(selected_ids) < 2:
            rnd = pick_random_id(fallback_cat)
            if rnd and rnd not in selected_ids:
                selected_ids.append(rnd)

        primary_id, secondary_id = selected_ids[0], selected_ids[1]

        print("[agent_route] forced compare:", primary_id, secondary_id)
        return {
            "action": "compare",
            "primary_id": primary_id,
            "secondary_id": secondary_id,
            "category": fallback_cat,
            "reason": "사용자가 비교를 요청해서 두 작품을 선택했습니다.",
            "candidates": candidates,
        }

    # 3) TTS 요청이면 → 후보 1개 또는 랜덤 1개 선택
    if is_tts:
        if candidates:
            primary_id = candidates[0]["id"]
        else:
            primary_id = pick_random_id(fallback_cat)

        print("[agent_route] forced tts:", primary_id)
        return {
            "action": "tts",
            "primary_id": primary_id,
            "secondary_id": None,
            "category": fallback_cat,
            "reason": "사용자가 음성 설명을 요청해서 해당 작품으로 TTS 모드를 선택했습니다.",
            "candidates": candidates,
        }

    # 4) 여기까지 왔는데 후보가 하나도 없으면 그냥 랜덤 curate
    if not candidates:
        rnd_id = pick_random_id(fallback_cat)
        return {
            "action": "curate",
            "primary_id": rnd_id,
            "secondary_id": None,
            "category": fallback_cat,
            "reason": "의미검색 결과가 없어, 임의의 작품을 추천했습니다.",
            "candidates": [],
        }

    # 5) 나머지 일반 케이스는 LLM에게 맡김 (기존 로직 그대로)
    prompt_lines = [
        "당신은 미술관 AI 서비스의 '라우팅 에이전트'입니다.",
        "사용자의 한 줄 요청을 보고, 아래 세 가지 중 어떤 기능으로 보내면 좋을지 결정하세요.",
        "",
        "1) 'curate': 특정 작품 하나에 대한 큐레이터 설명을 보여주는 화면 (상세 화면)으로 보냄.",
        "2) 'compare': 두 작품을 나란히 비교해 주는 화면으로 보냄.",
        "3) 'tts': 작품 하나의 설명을 들려주는 TTS 중심 화면으로 보냄. (실제 라우팅은 상세 화면과 동일).",
        "",
        "반드시 아래 JSON 형식만, 순수 JSON으로 출력하세요.",
        "{",
        '  "action": "curate" | "compare" | "tts",',
        '  "primary_id": "후보 목록 중 선택한 첫 번째 작품 id 또는 null",',
        '  "secondary_id": "비교가 필요한 경우 두 번째 작품 id, 아니면 null",',
        '  "category": "painting_json" | "craft_json" | "sculpture_json" | null,',
        '  "reason": "왜 이런 선택을 했는지 한국어로 한두 문장 설명"',
        "}",
        "",
        "규칙:",
        "- 반드시 아래 'candidate_artworks' 목록 안에 있는 id만 선택하세요.",
        "- 사용자가 '비교', '두 작품', 'vs', '차이' 등을 언급하면 action은 가급적 'compare'를 사용하세요.",
        "- 사용자가 '읽어줘', '설명 들어보고 싶어', '음성', 'tts' 등을 언급하면 action은 'tts'를 사용하세요.",
        "- 그 외의 경우는 기본값으로 'curate'를 사용하세요.",
        "- category는 특별히 언급이 없으면 null로 두어도 됩니다.",
        "",
        f"사용자 질의: {q}",
        "",
        "candidate_artworks:",
        json.dumps(candidates, ensure_ascii=False),
    ]
    prompt = "\n".join(prompt_lines)

    try:
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent generation failed: {e}")

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "action": "curate",
            "primary_id": candidates[0]["id"],
            "secondary_id": None,
            "category": fallback_cat,
            "reason": "LLM 응답을 파싱하지 못해 첫 번째 후보를 기본 추천으로 사용했습니다.",
        }

    action = parsed.get("action") or "curate"
    primary_id = parsed.get("primary_id")
    secondary_id = parsed.get("secondary_id")
    category = parsed.get("category") or fallback_cat
    reason = parsed.get("reason") or "사용자 요청에 따라 자동으로 선택했습니다."

    if not primary_id and candidates:
        primary_id = candidates[0]["id"]

    return {
        "action": action,
        "primary_id": primary_id,
        "secondary_id": secondary_id,
        "category": category,
        "reason": reason,
        "candidates": candidates,
    }

# ───────────────────────────────────────────────────────────
# 비교문 생성 엔드포인트
# ───────────────────────────────────────────────────────────
@app.post("/ai/analyze-compare")
async def analyze_compare(req: CompareIn):
    """
    두 작품 ID를 받아 비교 큐레이션 텍스트를 생성.
    프론트에서는 예를 들어:
      POST /ai/analyze-compare
      {
        "ids": ["idA", "idB"],
        "category": "painting_json",
        "locale": "ko"
      }
    형태로 호출.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")

    if len(req.ids) != 2:
        raise HTTPException(status_code=400, detail="ids는 정확히 2개가 필요합니다.")

    id_a, id_b = req.ids[0], req.ids[1]

    # 1) 카드 로드 (json_extracted에서)
    card_a = load_card_by_id(req.category, id_a)
    card_b = load_card_by_id(req.category, id_b)

    # 2) 두 카드 정보를 합쳐 RAG 질의 생성
    query_parts = [build_query(card_a), build_query(card_b)]
    query = "\n\n".join([q for q in query_parts if q])

    hits = retrieve_context(query, k=3)
    context_block = format_context(hits)

    # 3) 비교용 프롬프트 구성
    prompt_lines = [
        "당신은 국공립 미술관의 전문 큐레이터입니다.",
        "두 작품을 나란히 본 관람객에게, 편안한 한국어 구어체로 비교 감상을 도와주세요.",
        "",
        "### [작품 A]",
        json.dumps(card_a, ensure_ascii=False),
        "",
        "### [작품 B]",
        json.dumps(card_b, ensure_ascii=False),
        "",
        "### [검색 컨텍스트]",
        context_block,
        "",
        "설명은 3~5문단, 총 8~12문장 정도로 작성합니다.",
        "1) 두 작품의 공통된 주제나 분위기를 먼저 짚고,",
        "2) 표현 방식·재료·구도 등에서의 차이점을 자연스럽게 설명한 뒤,",
        "3) 관람자가 두 작품을 함께 보며 느껴볼 수 있는 감상 포인트를 제안해주세요.",
        "번호 매기기, 불릿 포인트, 이모지는 사용하지 마세요.",
        "권리·라이선스·데이터셋 출처 등 메타 정보는 언급하지 마세요.",
        "",
        "최종 출력에는 위의 섹션 제목을 포함하지 말고, 순수 단락 텍스트만 제시하세요.",
    ]
    prompt = "\n".join(prompt_lines)

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    def to_brief(card: Dict, fallback_id: str) -> Dict:
        # card 안에서 여러 후보 키 중 첫 번째로 값이 있는 것을 골라주는 헬퍼
        def first(*keys):
            for k in keys:
                v = card.get(k)
                if v not in (None, "", []):
                    return v
            return None

        return {
            "id": card.get("id") or fallback_id,
            # 제목
            "title": first(
                "title",
                "title_kor", "title_kr", "title_ko",
                "title_eng", "title_en",
            ),
            # 작가
            "artist": first(
                "artist",
                "artist_kor", "artist_kr", "artist_ko",
                "artist_eng", "artist_en",
            ),
            # 분류
            "class": first(
                "class",
                "class_kor", "class_kr", "class_ko",
                "class_eng", "class_en",
            ),
            # 연도/시기
            "year": first(
                "year",
                "date_or_period",
                "photo_date",
            ),
            # 재질
            "material": first(
                "material",
                "material_kor", "material_kr", "material_ko",
                "material_eng", "material_en",
            ),
        }

    return {
        "left": card_a,   # 요약본 대신 원본 카드 그대로
        "right": card_b,
        "analysis": text,
        "retrieved": [
            {"meta": h.get("meta"), "score": h.get("score"), "id": h.get("id")}
            for h in hits
        ],
    }


# ───────────────────────────────────────────────────────────
# Google Cloud TTS 엔드포인트
# ───────────────────────────────────────────────────────────
@app.post("/ai/tts")
async def tts_route(req: TtsIn):
    """
    설명 텍스트를 받아 Google Cloud TTS로 MP3 음성을 생성해서
    base64 문자열로 반환.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text가 비어 있습니다.")

    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=req.language_code or "ko-KR",
            name=req.voice_name or "ko-KR-Standard-A",  # 기본 한국어 음성
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=req.speaking_rate or 1.0,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        audio_b64 = base64.b64encode(response.audio_content).decode("utf-8")
        return {"audio_b64": audio_b64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 실패: {e}")
