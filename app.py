import os
import json
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ───────────────────────────────────────────────────────────
# 환경설정
# ───────────────────────────────────────────────────────────
load_dotenv()  # .env 지원

import google.generativeai as genai

API_KEY = os.environ.get("GOOGLE_API_KEY")            # 필수
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-004")

if API_KEY:
    genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ───────────────────────────────────────────────────────────
# Chroma(벡터DB) + Gemini 임베딩 설정
# ───────────────────────────────────────────────────────────
# ※ 사전 색인(build_index.py)으로 ./chroma_db 에 컬렉션을 만들어두는 것을 권장
#    없더라도 코드가 안전하게 동작(검색 미사용)하도록 예외 처리함
use_retriever = True
retrieval = None
try:
    import chromadb
    from chromadb import Settings
    from chromadb.utils.embedding_functions import EmbeddingFunction

    class GeminiEF(EmbeddingFunction):
        def __call__(self, texts: List[str]) -> List[List[float]]:
            out: List[List[float]] = []
            for t in texts:
                r = genai.embed_content(model=EMBED_MODEL, content=t)
                out.append(r["embedding"])
            return out

    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    retrieval = client.get_or_create_collection(
        name="curator_corpus",
        embedding_function=GeminiEF(),
        metadata={"hnsw:space": "cosine"},
    )
except Exception as e:
    # 벡터DB 초기화 실패 시에도 서비스는 생성만 가능하도록 유지
    use_retriever = False
    retrieval = None
    # 필요하면 로그로 남기세요
    # print(f"[WARN] Retriever disabled: {e}")

# ───────────────────────────────────────────────────────────
# FastAPI 앱
# ───────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────────────────
# 모델
# ───────────────────────────────────────────────────────────
class CurateIn(BaseModel):
    id: str
    card: Optional[Dict] = None  # py3.9 호환

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
        lines.append("")  # 빈 줄
    return "\n".join(lines)

# ───────────────────────────────────────────────────────────
# 프롬프트 빌더 (컨텍스트 흡수 버전)
# ───────────────────────────────────────────────────────────
def build_prompt(card: Dict, context_block: str) -> str:
    title = (card or {}).get("title") or (card or {}).get("title_ko") or (card or {}).get("title_en") or (card or {}).get("id", "")
    artist = (card or {}).get("artist") or (card or {}).get("artist_ko") or (card or {}).get("artist_en", "")
    klass = (card or {}).get("class") or (card or {}).get("class_ko") or (card or {}).get("class_en", "")
    cats = " / ".join((card or {}).get("categories", []) or [])
    material = (card or {}).get("material") or (card or {}).get("material_ko") or (card or {}).get("material_en", "")
    year = (card or {}).get("date_or_period") or (card or {}).get("photo_date", "")

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
    "최종 출력에는 위의 메타/지침 섹션을 포함하지 말고, 단락 텍스트만 제시하세요."
]
    return "\n".join(lines)



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

@app.post("/curate")
async def curate(req: CurateIn):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")

    card = req.card or {}

    # 1) 질의 조립
    query = build_query(card)

    # 2) 의미 검색 (리트리버가 켜져 있으면)
    hits = retrieve_context(query, k=5)

    # 3) 증강 프롬프트
    context_block = format_context(hits)
    prompt = build_prompt(card, context_block)

    # 4) 생성
    try:
        resp = model.generate_content(prompt)  # sync 호출
        text = (resp.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return {
        "curator_text": text,
        "retrieved": [
            {"meta": h.get("meta"), "score": h.get("score"), "id": h.get("id")}
            for h in hits
        ],  # 근거 확인용(원하면 프론트에서 숨겨도 됨)
    }

# (선택) 검색만 단독 제공하고 싶다면 간단 엔드포인트 추가
@app.get("/search")
def search(q: str, k: int = 5):
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

