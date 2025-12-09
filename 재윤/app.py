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
import asyncio
import re

# 🆕 CLIP / torch
import torch
import open_clip  # pip install open_clip_torch

# 🆕 chromadb
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
# 이미지 인덱스 로드 (prefix -> 상대경로)
# ───────────────────────────────────────────────────────────
IMAGE_INDEX_PATH = DATA_ROOT / "image_index.json"

try:
    with IMAGE_INDEX_PATH.open("r", encoding="utf-8") as f:
        IMAGE_INDEX = json.load(f)
    print(f"[IMAGE_INDEX] loaded {len(IMAGE_INDEX)} items from {IMAGE_INDEX_PATH}")
except FileNotFoundError:
    IMAGE_INDEX = {}
    print(f"[IMAGE_INDEX] NOT FOUND: {IMAGE_INDEX_PATH}, using empty index")

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

# ✅ chroma 클라이언트 & 컬렉션 설정 (텍스트 / 이미지 분리)
try:
    # 텍스트 임베딩 DB
    TEXT_CHROMA_PATH = r"C:\Exhibit\curator_server\backend\chroma_db_text"
    text_client = chromadb.PersistentClient(
        path=TEXT_CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    retrieval = text_client.get_or_create_collection(
        name="curator_corpus",
        embedding_function=GeminiEF(),          # 텍스트는 GeminiEF로 쿼리
        metadata={"hnsw:space": "cosine"},
    )

    # 이미지 임베딩 DB
    IMAGE_CHROMA_PATH = r"C:\Exhibit\chroma_db"
    image_client = chromadb.PersistentClient(
        path=IMAGE_CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    image_collection = image_client.get_or_create_collection(
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
# 캐시 (프로세스 메모리 기반)
# ───────────────────────────────────────────────────────────
# 같은 작품을 다시 열 때 Gemini를 다시 부르지 않도록 하는 캐시
CURATION_CACHE: Dict[str, str] = {}          # id -> curator_text
IMMERSIVE_CACHE: Dict[str, Dict] = {}        # id -> {"text": str, "labels": List[str]}

# 같은 텍스트에 대해 TTS를 다시 호출하지 않도록 하는 캐시
from hashlib import md5
TTS_CACHE: Dict[str, str] = {}               # key -> audio_b64

# 🔑 Immersive 캐시 키 헬퍼
def make_immersive_key(card_id: Optional[str], category: Optional[str]) -> Optional[str]:
    if not card_id:
        return None
    return f"{category or 'any'}::{card_id}"

# 🆕 에이전트 캐시
AGENT_CACHE: Dict[str, Dict] = {}    

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

class CurateImmersiveIn(BaseModel):  # 👈 여기 추가
    id: Optional[str] = None
    category: Optional[str] = None
    card: Optional[Dict] = None

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
# 제목/작가 기반 검색 유틸
# ───────────────────────────────────────────────────────────
def extract_title_from_query(q: str) -> Optional[str]:
    """
    '의기라는 작품 보여줘', '의기라는 작품 들려줘' 같이
    '~라는 작품 ...' 패턴에서 제목 부분만 뽑아낸다.
    """
    if not q:
        return None
    q = q.strip()

    # 1) 가장 확실한 패턴: '라는 작품'
    if "라는 작품" in q:
        before = q.split("라는 작품")[0].strip()
        return before or None

    # 2) '작품 보여줘', '작품 들려줘'만 있는 경우까지 커버
    for key in ["작품 보여줘", "작품 들려줘"]:
        if key in q:
            before = q.split(key)[0].strip()
            # 조사 정리
            for josa in ["를", "을", "은", "는", "이", "가"]:
                if before.endswith(josa):
                    before = before[:-1]
            return before or None

    return None

def extract_two_titles_from_query(q: str) -> List[str]:
    """
    '의기와 빨래터라는 작품 두 개 비교해줘',
    '의기랑 빨래터 작품 비교해줘' 같은 문장에서
    제목 후보 두 개를 뽑는다.
    """
    if not q:
        return []
    q = q.strip()

    # 1) 가장 자주 쓸만한 패턴들
    patterns = [
        r"(.+?)(?:와|과|이랑|랑)\s*(.+?)(?:라는 작품|작품 두 개|작품을 두 개|작품 비교)",
        r"(.+?)(?:와|과|이랑|랑)\s*(.+?)\s*작품\s*비교",
    ]

    def clean(token: str) -> str:
        token = token.strip()
        for josa in ["를", "을", "은", "는", "이", "가", "의"]:
            if token.endswith(josa):
                token = token[:-1]
        return token.strip()

    for pat in patterns:
        m = re.search(pat, q)
        if m:
            left = clean(m.group(1))
            right = clean(m.group(2))
            result = [t for t in [left, right] if t]
            # 두 개 다 있으면 반환
            if len(result) >= 2:
                return result

    # 패턴에서 못 찾으면 그냥 비워 둠 (기존 로직 사용)
    return []

def find_cards_by_title_keyword(
    keyword: str,
    category: Optional[str] = "painting_json",
    max_results: int = 3,
) -> List[Dict]:
    """
    json_extracted/{category} 아래에서
    제목(한/영) 안에 keyword가 포함된 작품들을 찾는다.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    real_cat = map_category(category or "painting_json")
    target_dir = JSON_ROOT / real_cat
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    matches: List[Dict] = []

    for p in target_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                card = json.load(f)
        except Exception:
            continue

        desc = card.get("Description") or {}

        # 제목 후보 필드들
        title_candidates = [
            card.get("title"),
            card.get("title_kor"),
            card.get("title_ko"),
            card.get("title_eng"),
            desc.get("ArtTitle_kor"),
            desc.get("ArtTitle_eng"),
        ]

        hit = False
        for t in title_candidates:
            if t and keyword in str(t):
                hit = True
                break

        if not hit:
            continue

        matches.append(
            {
                "id": p.stem,
                "title": next((t for t in title_candidates if t), ""),
            }
        )

        if len(matches) >= max_results:
            break

    return matches

def extract_artist_from_query(q: str) -> Optional[str]:
    """
    '김환기 작가의 작품 보여줘', '박수근 작가 작품 들려줘'
    같은 문장에서 '작가' 앞에 있는 이름만 뽑기.
    """
    if not q:
        return None
    q = q.strip()

    # 1) '작가의 작품' 패턴
    if "작가의 작품" in q:
        before = q.split("작가의 작품")[0].strip()
    # 2) '작가 작품' 패턴
    elif "작가 작품" in q:
        before = q.split("작가 작품")[0].strip()
    # 3) '작가의' 만 있는 경우
    elif "작가의" in q:
        before = q.split("작가의")[0].strip()
    # 4) 그냥 '작가'만 있는 경우 (예: "김환기 작가 그림 들려줘")
    elif "작가" in q:
        before = q.split("작가")[0].strip()
    else:
        return None

    # 조사/공백 정리
    for josa in ["를", "을", "은", "는", "이", "가", "의"]:
        if before.endswith(josa):
            before = before[:-1]
    before = before.strip()

    return before or None

def find_cards_by_artist_keyword(
    keyword: str,
    category: Optional[str] = "painting_json",
    max_results: int = 5,
) -> List[Dict]:
    """
    json_extracted/{category} 아래에서
    작가명(한/영) 안에 keyword가 포함된 작품들을 찾는다.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    real_cat = map_category(category or "painting_json")
    target_dir = JSON_ROOT / real_cat
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    matches: List[Dict] = []

    for p in target_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                card = json.load(f)
        except Exception:
            continue

        desc = card.get("Description") or {}

        # 작가 후보 필드들
        artist_candidates = [
            card.get("artist"),
            card.get("artist_kor"),
            card.get("artist_ko"),
            card.get("artist_eng"),
            desc.get("ArtistName_kor"),
            desc.get("ArtistName_eng"),
        ]

        hit = False
        for a in artist_candidates:
            if a and keyword in str(a):
                hit = True
                break

        if not hit:
            continue

        matches.append(
            {
                "id": p.stem,
                "artist": next((a for a in artist_candidates if a), ""),
            }
        )

        if len(matches) >= max_results:
            break

    return matches


def extract_subject_from_painting_query(q: str) -> Optional[str]:
    """
    '참새가 그려진 작품 보여줘', '사람이 나오는 그림',
    '호박 있는 작품', '호박 그림 보여줘' 등에서
    핵심 대상을 뽑아낸다.
    """
    if not q:
        return None
    q = q.strip()

    # 1) 자주 나오는 패턴들
    patterns = [
        "가 그려진", "이 그려진",
        "가 나오는", "이 나오는",
        "가 있는",   "이 있는",
        "이 등장하는", "가 등장하는",
        "가 보이는", "이 보이는",
    ]

    for pat in patterns:
        if pat in q:
            before = q.split(pat)[0].strip()
            for josa in ["를", "을", "은", "는", "이", "가", "의"]:
                if before.endswith(josa):
                    before = before[:-1]
            before = before.strip()
            return before or None

    # 2) '~ 있는 그림/작품/사진' 형태 (조사 생략 버전)
    #    예: "호박 있는 작품 보여줘"
    for key in ["그림", "작품", "사진"]:
        if key in q and "있는" in q:
            # "호박 있는 작품" 에서 "있는" 앞 단어 하나만 잡기
            # ... "호박 있는 작품" → ["호박", "있는", "작품"]
            tokens = q.split()
            try:
                idx = tokens.index("있는")
                if idx > 0:
                    cand = tokens[idx - 1]
                    for josa in ["를", "을", "은", "는", "이", "가", "의"]:
                        if cand.endswith(josa):
                            cand = cand[:-1]
                    cand = cand.strip()
                    if cand:
                        return cand
            except ValueError:
                pass

    # 3) '호박 그림', '참새 그림', '바다 그림' 같은 단순형
    if "그림" in q or "작품" in q or "사진" in q:
        for key in ["그림", "작품", "사진"]:
            if key in q:
                before = q.split(key)[0].strip()
                tokens = before.split()
                if tokens:
                    cand = tokens[-1]
                    for josa in ["를", "을", "은", "는", "이", "가", "의"]:
                        if cand.endswith(josa):
                            cand = cand[:-1]
                    cand = cand.strip()
                    if cand:
                        return cand

    return None



def find_cards_by_caption_keyword(
    keyword: str,
    category: Optional[str] = "painting_json",
    max_results: int = 8,
) -> List[Dict]:
    """
    json_extracted/{category} 아래에서
    vision_caption_ko 안에 keyword가 포함된 작품들을 찾는다.
    (chroma_db_text를 다시 만들 필요 없이,
     JSON 파일만 직접 검색하는 방식)
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    real_cat = map_category(category or "painting_json")
    target_dir = JSON_ROOT / real_cat
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    matches: List[Dict] = []

    for p in target_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                card = json.load(f)
        except Exception:
            continue

        caption = (
            card.get("vision_caption_ko")
            or card.get("vision_caption")
            or card.get("vision_caption_en")
            or ""
        )

        if not caption:
            continue

        if keyword in str(caption):
            desc = card.get("Description") or {}
            title_candidates = [
                card.get("title"),
                card.get("title_kor"),
                card.get("title_ko"),
                card.get("title_eng"),
                desc.get("ArtTitle_kor"),
                desc.get("ArtTitle_eng"),
            ]
            matches.append(
                {
                    "id": p.stem,
                    "title": next((t for t in title_candidates if t), ""),
                }
            )

            if len(matches) >= max_results:
                break

    return matches


# ───────────────────────────────────────────────────────────
# RAG 유틸
# ───────────────────────────────────────────────────────────
def build_query(card: Dict) -> str:
    """카드의 핵심 필드 + AiCaption으로 의미검색용 질의문을 조립"""
    caption = (
        card.get("vision_caption_ko")
        or card.get("vision_caption")
        or card.get("vision_caption_en")
        or ""
    )

    parts: List[Optional[str]] = [
        card.get("title") or card.get("title_ko") or card.get("title_en"),
        card.get("artist") or card.get("artist_ko") or card.get("artist_en"),
        card.get("class") or card.get("class_ko") or card.get("class_en"),
        " ".join(card.get("categories", []) or []),
        card.get("material") or card.get("material_ko") or card.get("material_en"),
        card.get("date_or_period") or card.get("photo_date"),
        caption,  # 👈 여기!
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
        score = dists[i] if i < len(dists) else None
        # 🔥 numpy.float32 → 파이썬 float로 강제 캐스팅
        if score is not None:
            try:
                score = float(score)
            except Exception:
                score = None

        hits.append({
            "id": ids[i],
            "text": docs[i] if i < len(docs) else "",
            "meta": metas[i] if i < len(metas) else {},
            "score": score,
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

def to_clip_query(q: str) -> str:
    """
    한국어(또는 자연어) 검색 문장을
    CLIP이 이해하기 좋은 짧은 영어 시각 묘사로 바꿔준다.
    예) '참새가 그려진 작품 보여줘' -> 'a painting of a sparrow'
    """
    q = (q or "").strip()
    if not q:
        return q

    # 이미 영어가 섞인 경우는 그대로 써도 무방하지만,
    # 여기서는 일단 LLM 번역 한 번 태우는 방식으로 단순화.
    prompt = (
        "You are a helper that converts Korean art search queries "
        "into short English visual descriptions suitable for CLIP text encoder.\n"
        "Examples:\n"
        "1) '참새가 그려진 작품 보여줘' -> 'a painting of a sparrow'\n"
        "2) '바닷가 풍경 그림 보여줘' -> 'a painting of a seaside landscape'\n"
        "3) '밤하늘에 별이 많은 그림' -> 'a painting of a starry night sky'\n"
        "Only output the final English phrase, no quotes, no extra text.\n\n"
        f"Query: {q}\n"
        "English visual description:"
    )

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        # 혹시 이상하게 나오면 원문 fallback
        if not text:
            return q
        # 너무 길면 CLIP에 안 좋으니 30~40단어 정도까지만 잘라줌 (옵션)
        words = text.split()
        if len(words) > 40:
            text = " ".join(words[:40])
        return text
    except Exception as e:
        print("[to_clip_query] translation failed, fallback to original:", e)
        return q


def retrieve_image_context(query: str, k: int = 5) -> List[Dict]:
    """
    CLIP 텍스트 임베딩으로 curator_image_clip에서
    '이미지 기반' Top-k 작품을 찾음.
    반환 형식은 텍스트 RAG와 비슷하게 맞춰서 사용하기 쉽게.
    """
    if not use_image_retriever or not image_collection or not query:
        return []

    # 🔥 1단계: 한국어 쿼리를 CLIP용 짧은 영어 묘사로 변환
    clip_query = to_clip_query(query)
    print(f"[retrieve_image_context] raw query='{query}' -> clip_query='{clip_query}'")

    try:
        # 🔥 한글 대신 변환된 영어 텍스트로 CLIP 임베딩 생성
        vec = embed_clip_text([clip_query])[0]  # 1개 쿼리 → 1벡터
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
        score = dists[i] if i < len(dists) else None
        if score is not None:
            try:
                score = float(score)
            except Exception:
                score = None

        hits.append(
            {
                "id": ids[i],
                "meta": metas[i] if i < len(metas) else {},
                "score": score,
            }
        )
    return hits


def _to_image_url(raw_path: Optional[str]) -> Optional[str]:
    """
    chroma 메타에 저장된 로컬 경로(D:\\Exhibit\\image_extracted\\...)를
    프론트에서 쓸 수 있는 URL(/image_extracted/...)로 변환
    """
    if not raw_path:
        return None

    p = Path(raw_path)
    try:
        # IMG_ROOT = DATA_ROOT / "image_extracted" 위에서 정의됨
        rel = p.relative_to(IMG_ROOT)
        return f"/image_extracted/{rel.as_posix()}"
    except Exception:
        # 혹시 이미 /image_extracted/... 형식이라면 그대로 사용
        s = str(raw_path).replace("\\", "/")
        if s.startswith("/image_extracted/"):
            return s
        return None

# ─────────────────────────────────────────────
# CLIP 기반 유사 이미지 검색 (에러 나도 500 안 던지게)
# ─────────────────────────────────────────────
def similar_images_by_id(
    base_id: str,
    k: int = 5,
    category: Optional[str] = None,
) -> List[Dict]:
    """
    curator_image_clip 컬렉션 안에서
    - base_id 작품과 CLIP 기준으로 비슷한 작품 k개 찾기
    - category가 주어지면 같은 category만 필터링 (painting_json / craft_json / sculpture_json 등)

    ⚠ 에러가 나더라도 HTTPException 안 던지고, 그냥 [] 리턴해서
      프론트에서는 "유사한 작품을 찾지 못했습니다." 로 처리되게 만든다.
    """
    if not use_image_retriever or image_collection is None:
        print("[similar_images_by_id] image retriever disabled")
        return []

    # 1) 기준 작품의 embedding 꺼내기
    try:
        doc = image_collection.get(
            ids=[base_id],
            include=["embeddings", "metadatas"],
        )
    except Exception as e:
        print(f"[similar_images_by_id] get() error for id={base_id} -> {e}")
        return []

    embeddings = doc.get("embeddings")
    if embeddings is None:
        print(f"[similar_images_by_id] no embeddings field for id={base_id}")
        return []

    # Chroma가 numpy array로 줄 수도 있어서 list로 강제 변환
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    # 보통 [[...]] 형태라 첫 번째 요소 꺼냄
    if len(embeddings) == 0:
        print(f"[similar_images_by_id] empty embeddings for id={base_id}")
        return []

    base_emb = embeddings[0]
    if hasattr(base_emb, "tolist"):
        base_emb = base_emb.tolist()

    base_meta_list = doc.get("metadatas") or [{}]
    base_meta = base_meta_list[0] if base_meta_list else {}

    # 2) 이 embedding으로 근접 이웃 검색
    try:
        res = image_collection.query(
            query_embeddings=[base_emb],
            n_results=k + 10,  # 자기 자신 + 카테고리 필터 고려해서 여유 있게
        )
    except Exception as e:
        print(f"[similar_images_by_id] query() error for id={base_id} -> {e}")
        return []

    raw_ids = res.get("ids")
    if raw_ids is None or len(raw_ids) == 0:
        return []

    ids = raw_ids[0]
    if not isinstance(ids, list):
        ids = list(ids)

    raw_metas = res.get("metadatas")
    if raw_metas is None or len(raw_metas) == 0:
        metas = [{} for _ in ids]
    else:
        metas = raw_metas[0]
        if not isinstance(metas, list):
            metas = list(metas)

    raw_dists = res.get("distances")
    if raw_dists is None or len(raw_dists) == 0:
        dists = [None] * len(ids)
    else:
        dists = raw_dists[0]
        if not isinstance(dists, list):
            dists = list(dists)

    items: List[Dict] = []

    for i, cid in enumerate(ids):
        # 자기 자신은 제외
        if cid == base_id:
            continue

        meta = metas[i] if i < len(metas) else {}
        score = dists[i] if i < len(dists) else None

        # numpy.float32 → float
        if score is not None:
            try:
                score = float(score)
            except Exception:
                score = None

        # category 필터
        if category:
            m_cat = meta.get("category")
            if m_cat is not None and m_cat != category:
                continue

        # 🔥 로컬 경로 → URL 변환
        raw_img_path = meta.get("image_path")
        img_url = _to_image_url(raw_img_path)

        # 🔁 메타에 경로가 없거나 변환 실패하면, id(prefix)로 인덱스에서 찾기 (백업)
        if not img_url:
            rel = IMAGE_INDEX.get(cid)
            if rel:
                img_url = f"/image_extracted/{rel}"

        items.append(
            {
                "id": cid,
                "title": meta.get("title", ""),
                "artist": meta.get("artist", ""),
                "class": meta.get("class", ""),
                "year": meta.get("year", ""),
                "category": meta.get("category"),       # ex) "painting_json"
                # 프론트에서 바로 `${API}${item.image_path}`로 사용할 수 있는 형태
                "image_path": img_url,
                "score": score,
            }
        )

        if len(items) >= k:
            break

    return items

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

def build_immersive_prompt(
    card: Dict,
    context_block: str,
    visual_labels: List[str],
) -> str:
    card = card or {}
    desc = card.get("Description") or {}
    photo = card.get("Photo_Info") or {}
    data_info = card.get("Data_Info") or {}

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
        or desc.get("Period_kor")
        or desc.get("Period_eng")
        or ""
    )

    meta_targets: List[str] = []
    for s in [
        title,
        klass,
        desc.get("Subject_kor"),
        desc.get("Subject_eng"),
        desc.get("Keyword_kor"),
        desc.get("Keyword_eng"),
    ]:
        if s:
            meta_targets.append(str(s))

    allowed_targets_list = list(dict.fromkeys((visual_labels or []) + meta_targets))
    labels_str = (
        ", ".join(allowed_targets_list) if allowed_targets_list else "특별히 추출된 단어 없음"
    )

    _ = context_block

    lines: List[str] = [
        "당신은 한국어로 해설하는 미술관 도슨트입니다.",
        "",
        "[작품 기본 정보]",
        f"제목: {title}",
        f"작가: {artist}",
        f"분류/장르: {klass}",
        f"재질: {material}",
        f"연도/시기: {year}",
        "",
        "[이 작품과 직접 관련된 단어 목록]",
        labels_str,
        "",
        "위 목록에는 작품 제목·주제·이미지 분석을 통해 얻은 단어들만 들어 있습니다.",
        "구체적인 사물 이름(예: 새, 꽃, 금붕어, 사람, 글씨 등)을 말할 때는 가급적 이 목록 안에 실제로 적힌 단어를 그대로 사용하십시오.",
        "목록에 없는 전혀 다른 사물을 상상해서 새로 추가하지 마십시오.",
        "",
        "지침:",
        "1. 전체 해설을 정확히 3개의 문단으로 작성합니다.",
        "   - 첫 번째 문단: 작품 전체 분위기와 의미를 2~3문장으로 소개합니다.",
        "   - 두 번째 문단: 화면 왼쪽, 화면 가운데, 화면 오른쪽을 이 순서대로 언급하지만, 구체적인 사물의 위치를 추측하지 말고, 구도와 시선 흐름을 추상적으로 설명합니다.",
        "   - 세 번째 문단: 관람자가 느끼면 좋을 감정·메시지·여운을 2~3문장으로 정리합니다.",
        "2. 첫 번째 문단과 세 번째 문단에서는 '화면 왼쪽', '화면 가운데', '화면 오른쪽', '왼쪽', '가운데', '오른쪽' 같은 방향 표현을 사용하지 마십시오.",
        "3. 두 번째 문단에서는 반드시 다음 표현을 이 순서대로 한 번씩만 사용합니다: '화면 왼쪽에는', '화면 가운데에는', '화면 오른쪽에는'.",
        "4. 그러나 두 번째 문단에서도 '호박이 화면 왼쪽에 있다', '글씨가 화면 오른쪽에 있다'처럼 특정 사물과 정확한 위치를 결합해서 말하지 마십시오.",
        "   대신 '화면 왼쪽에는 조용한 여백이 펼쳐지고', '화면 가운데에는 시선이 머무는 중심 부분이 자리하며', '화면 오른쪽에는 전체 분위기를 정리하는 요소들이 놓여 있는 듯합니다'처럼 추상적인 표현을 사용하십시오.",
        "5. 글자·문장·주기도문·서예와 관련된 설명을 할 때에는, 그 글자가 화면의 어느 쪽에 있는지 단정해서 말하지 말고 위치 표현 없이 설명하십시오.",
        "6. 한 문단 안에서는 줄바꿈을 하지 말고, 자연스럽게 한 문단으로 이어서 작성합니다.",
        "7. 각 문단이 끝날 때마다 한 줄을 완전히 비우고, 다음 문단을 새 줄에서 시작하십시오. (즉, 문단 사이에 빈 줄 한 줄을 넣으십시오.)",
        "8. 번호, 불릿, 큰따옴표는 출력에 포함하지 말고, 오직 세 개의 문단 텍스트만 출력하십시오.",
    ]

    if any("새" in t for t in allowed_targets_list):
        lines.append(
            "9. 목록에 '새'가 있다면, 적어도 한 문장에서는 새의 모습이나 느낌을 구체적으로 설명하되, 화면의 정확한 위치는 추측하지 마십시오."
        )
    if any("물고기" in t or "금붕어" in t for t in allowed_targets_list):
        lines.append(
            "10. 목록에 '물고기'나 '금붕어'가 있다면, 적어도 한 문장에서는 물고기의 색감이나 움직임을 설명하되, 화면의 정확한 위치는 추측하지 마십시오."
        )
    if any("사람" in t for t in allowed_targets_list):
        lines.append(
            "11. 목록에 '사람'이 있다면, 인물의 자세나 표정을 한 문장 이상에서 설명하되, 화면의 어느 쪽에 있는지 단정하지 마십시오."
        )

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

def filter_candidates_by_category(cands, category):
    """
    에이전트 후보 중에서,
    주어진 category에서 JSON 카드가 실제로 존재하는 id만 남긴다.
    """
    if not category:
        return cands

    valid = []
    for c in cands:
        cid = c.get("id")
        if not cid:
            continue
        try:
            # 파일이 있으면 그대로 통과
            _ = load_card_by_id(category, cid)
            valid.append(c)
        except HTTPException:
            # 카드 없는 id는 버림
            continue
    return valid

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
    image_index.json에서 prefix에 해당하는 상대 경로를 찾아
    /image_extracted/... 형태의 URL로 반환.
    """
    rel = IMAGE_INDEX.get(prefix)
    if not rel:
        raise HTTPException(status_code=404,
                            detail=f"image not found for prefix={prefix}")

    url = f"/image_extracted/{rel}"
    return {"url": url}

@app.post("/curate")
async def curate(req: CurateIn):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")

    card = req.card or {}
    card_id = req.id or card.get("id")

    # 🔍 디버그 로그 추가
    print(f"[curate] request id={card_id}")

    # 1) 캐시 먼저 확인
    if card_id and card_id in CURATION_CACHE:
        print(f"[curate] CACHE HIT for {card_id}")
        return {
            "curator_text": CURATION_CACHE[card_id],
            "retrieved": [],
        }

    print(f"[curate] CACHE MISS for {card_id}")

    # 2) 평소처럼 RAG + Gemini 호출
    query = build_query(card)
    hits = retrieve_context(query, k=5)
    context_block = format_context(hits)
    prompt = build_prompt(card, context_block)

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # 3) 캐시에 저장
    if card_id:
        CURATION_CACHE[card_id] = text

    return {
        "curator_text": text,
        "retrieved": [
            {"meta": h.get("meta"), "score": h.get("score"), "id": h.get("id")}
            for h in hits
        ],
    }



@app.post("/curate/immersive")
async def curate_immersive(req: CurateImmersiveIn):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")
    """
    몰입형(Immersive) 전용 3문단 해설 생성 엔드포인트.
    Immersive.jsx에서 POST /curate/immersive 로 호출함.
    """
    try:
        # 1) 카드 준비
        card = req.card or {}
        if not card and req.id:
            try:
                card = load_card_by_id(req.category, req.id)
            except Exception:
                return {
                    "curator_text": "작품 정보를 불러오지 못했습니다.",
                    "labels": [],
                }

        # 🔑 이 작품을 구분할 id + 카테고리까지 묶어서 캐시 키 생성
        card_id = req.id or card.get("id")
        cache_key = make_immersive_key(card_id, req.category)

        # 2) 캐시 먼저 확인 (카테고리까지 포함된 키 사용)
        if cache_key and cache_key in IMMERSIVE_CACHE:
            cached = IMMERSIVE_CACHE[cache_key]
            return {
                "curator_text": cached.get("text", ""),
                "labels": cached.get("labels", []),
            }

        # 3) (선택) 이미지 분석 라벨 - 현재는 CLIP 안 쓰고 빈 리스트만 사용
        visual_labels: List[str] = []

        # 4) 몰입형 전용 프롬프트 생성
        prompt = build_immersive_prompt(card, "", visual_labels)

        # 5) Gemini 호출을 별도 스레드에서 실행 + 타임아웃
        async def _gemini():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt),
            )

        try:
            resp = await asyncio.wait_for(_gemini(), timeout=15)
            text = (resp.text or "").strip()

        except asyncio.TimeoutError:
            print("❌ Gemini TIMEOUT (immersive)")
            text = (
                "이 작품은 전체적으로 고요하면서도 생동감 있는 분위기를 담고 있습니다. "
                "화면을 천천히 훑어 보며 색감과 구도를 감상해 보세요.\n\n"
                "화면 왼쪽에는, 화면 가운데에는, 화면 오른쪽에는 각각 다른 요소들이 자리하고 있으니 "
                "차례로 시선을 옮겨 보시기 바랍니다.\n\n"
                "지금은 AI 해설이 완전히 생성되지 않았지만, 작품이 주는 여운과 감정을 천천히 느껴 보세요."
            )

        except Exception as e:
            print("❌ Gemini ERROR (immersive):", e)
            text = (
                "현재 AI 해설을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요.\n\n"
                "그동안에는 화면 전체를 천천히 살펴보며, 색감과 구도, 등장하는 대상들을 직접 감상해 보시길 권합니다."
            )

        # 6) 캐시에 저장 (여기서도 cache_key 사용)
        if cache_key:
            IMMERSIVE_CACHE[cache_key] = {
                "text": text,
                "labels": visual_labels,
            }

        return {
            "curator_text": text,
            "labels": visual_labels,
        }

    except Exception as e:
        print("❌ IMMERSIVE UNKNOWN ERROR:", e)
        return {
            "curator_text": "해설을 불러오는 중 문제가 발생했습니다.",
            "labels": [],
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
    results = []

    for h in hits:
        meta = (h.get("meta") or {})
        raw_img = meta.get("image_path")
        img_url = _to_image_url(raw_img)

        # 그래도 없으면 id(prefix)로 인덱스에서 찾기 (백업)
        if not img_url:
            cid = h.get("id")
            if cid:
                rel = IMAGE_INDEX.get(cid)
                if rel:
                    img_url = f"/image_extracted/{rel}"

        results.append(
            {
                "id": h.get("id"),
                "title": meta.get("title"),
                "artist": meta.get("artist"),
                "class": meta.get("class"),
                "year": meta.get("year"),
                "image_path": img_url,
                "score": h.get("score"),
            }
        )

    return {"results": results}

# ───────────────────────────────────────────────────────────
# 에이전트 라우트
# ───────────────────────────────────────────────────────────
@app.post("/ai/agent")
async def agent_route(req: AgentIn):
    """
    첫 화면(Welcome)에서 자연어 한 줄 입력을 받아
    - action: "curate" | "compare" | "tts"
    - primary_id / secondary_id
    - category
    를 판단해서 반환하는 라우터.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")

    # 기본 카테고리
    fallback_cat = req.category or "painting_json"

    # 🔑 에이전트 캐시 키 (카테고리 + 원문 질의)
    cache_key = f"{fallback_cat}::{q}"

    # 🔍 캐시 먼저 확인
    if cache_key in AGENT_CACHE:
        print(f"[agent_route] CACHE HIT for {cache_key}")
        return AGENT_CACHE[cache_key]

    print(f"[agent_route] CACHE MISS for {cache_key}")

    # -------------------------------
    # 0. 쿼리 전처리 및 플래그
    # -------------------------------
    lower_q = q.lower()
    compare_keywords = ["비교", "두 작품", "두 점", "2점", "vs", "차이"]
    tts_keywords = ["읽어줘", "읽어 줘", "설명 들어보고 싶어", "음성", "tts", "음성으로", "들려줘", "들려 줘"]

    is_compare = any(kw in lower_q for kw in compare_keywords)
    is_tts = any(kw in lower_q for kw in tts_keywords)

    # 👇 아래 있던 fallback_cat = ... 줄은 이제 위로 올라갔으니 삭제!
    # fallback_cat = req.category or "painting_json"


    # 제목 / 작가 / 주제 키워드 추출
    title_kw = extract_title_from_query(q)
    artist_kw = extract_artist_from_query(q)
    subject_kw = extract_subject_from_painting_query(q)

    # -------------------------------
    # 0-A. "제목 두 개" 패턴 처리
    # -------------------------------
    two_title_kws = extract_two_titles_from_query(q)
    multi_title_matches: List[Dict] = []

    if two_title_kws:
        for kw in two_title_kws:
            ms = find_cards_by_title_keyword(
                keyword=kw,
                category=fallback_cat,
                max_results=1,    # 각 제목당 1개
            )
            if ms:
                multi_title_matches.extend(ms)

        if multi_title_matches:
            print(
                "[agent_route] multi title match:",
                two_title_kws,
                "->",
                [m["id"] for m in multi_title_matches],
            )

    # -------------------------------
    # 0-B. 제목 직접 매칭
    # -------------------------------
    direct_title_matches: List[Dict] = []
    if title_kw:
        direct_title_matches = find_cards_by_title_keyword(
            keyword=title_kw,
            category=fallback_cat,
            max_results=3,
        )
        if direct_title_matches:
            print(
                "[agent_route] direct title match:",
                title_kw,
                "->",
                [m["id"] for m in direct_title_matches],
            )

    # -------------------------------
    # 0-C. 작가 직접 매칭
    # -------------------------------
    direct_artist_matches: List[Dict] = []
    if artist_kw:
        direct_artist_matches = find_cards_by_artist_keyword(
            keyword=artist_kw,
            category=fallback_cat,
            max_results=5,
        )
        if direct_artist_matches:
            print(
                "[agent_route] direct artist match:",
                artist_kw,
                "->",
                [m["id"] for m in direct_artist_matches],
            )

    # -------------------------------
    # 0-D. Vision 캡션(장면) 매칭
    # -------------------------------
    caption_matches: List[Dict] = []
    if subject_kw:
        caption_matches = find_cards_by_caption_keyword(
            keyword=subject_kw,
            category=fallback_cat,
            max_results=5,
        )
        if caption_matches:
            print(
                "[agent_route] caption match:",
                subject_kw,
                "->",
                [m["id"] for m in caption_matches],
            )

    # -------------------------------
    # 0-E. 비교/tts가 아닌 경우의 빠른 단일 추천
    # -------------------------------
    if direct_title_matches and not is_compare and not is_tts:
        primary_id = direct_title_matches[0]["id"]
        result = {
            "action": "curate",
            "primary_id": primary_id,
            "secondary_id": None,
            "category": fallback_cat,
            "reason": f"'{title_kw}'라는 작품 제목과 일치하는 작품을 직접 찾아 추천했습니다.",
            "candidates": direct_title_matches,
        }
        AGENT_CACHE[cache_key] = result
        return result


    if direct_artist_matches and not is_compare and not is_tts:
        primary_id = direct_artist_matches[0]["id"]
        result = {
            "action": "curate",
            "primary_id": primary_id,
            "secondary_id": None,
            "category": fallback_cat,
            "reason": f"'{artist_kw}' 작가의 작품 중 하나를 직접 찾아 추천했습니다.",
            "candidates": direct_artist_matches,
        }
        AGENT_CACHE[cache_key] = result
        return result

    if caption_matches and not is_compare and not is_tts:
        primary_id = caption_matches[0]["id"]
        result = {
            "action": "curate",
            "primary_id": primary_id,
            "secondary_id": None,
            "category": fallback_cat,
            "reason": f"'{subject_kw}'이(가) 들어간 장면이 Vision 캡션에 포함된 작품을 직접 찾아 추천했습니다.",
            "candidates": caption_matches,
        }
        AGENT_CACHE[cache_key] = result
        return result

    # -------------------------------
    # 1. 의미 검색 (텍스트 RAG + 이미지 RAG)
    # -------------------------------
    text_hits = retrieve_context(q, k=4)
    image_hits = retrieve_image_context(q, k=4)

    candidates: List[Dict] = []
    seen_ids = set()

    # 텍스트 RAG 결과
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

    # 이미지 RAG 결과
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
    print("[agent_route] candidates(before filter):", [c["id"] for c in candidates])

    # 실제 JSON이 있는 id만 남기기
    candidates = filter_candidates_by_category(candidates, fallback_cat)
    print("[agent_route] candidates(after filter):", [c["id"] for c in candidates])

    # -------------------------------
    # 2. 비교 요청
    # -------------------------------
    if is_compare:
        selected_ids: List[str] = []

        # 2-0) 제목 두 개 명시 → 우선
        if multi_title_matches:
            for m in multi_title_matches:
                cid = m.get("id")
                if cid and cid not in selected_ids:
                    selected_ids.append(cid)
                if len(selected_ids) >= 2:
                    break

        # 2-1) 작가 직접 매치 → 그 중 2점
        if len(selected_ids) < 2 and direct_artist_matches:
            for c in direct_artist_matches:
                cid = c.get("id")
                if cid and cid not in selected_ids:
                    selected_ids.append(cid)
                if len(selected_ids) >= 2:
                    break

        # 2-2) 그래도 부족하면 의미검색 후보에서 채우기
        if len(selected_ids) < 2:
            for c in candidates:
                cid = c.get("id")
                if cid and cid not in selected_ids:
                    selected_ids.append(cid)
                if len(selected_ids) >= 2:
                    break

        # 2-3) 그래도 모자라면 랜덤
        while len(selected_ids) < 2:
            rnd = pick_random_id(fallback_cat)
            if rnd and rnd not in selected_ids:
                selected_ids.append(rnd)

        primary_id, secondary_id = selected_ids[0], selected_ids[1]
        print("[agent_route] forced compare:", primary_id, secondary_id)

        if multi_title_matches:
            reason = (
                f"사용자가 제목으로 지목한 '{two_title_kws[0]}'와(과) "
                f"'{two_title_kws[1]}' 작품을 우선 선택해 비교합니다."
            )
        elif direct_artist_matches:
            reason = (
                f"사용자가 '{artist_kw}' 작가의 작품 비교를 요청해서 "
                "해당 작가의 작품을 우선적으로 두 점 선택했습니다."
            )
        else:
            reason = "사용자가 비교를 요청해서 관련성이 높은 두 작품을 선택했습니다."

        result = {
            "action": "compare",
            "primary_id": primary_id,
            "secondary_id": secondary_id,
            "category": fallback_cat,
            "reason": reason,
            "candidates": candidates,
        }
        AGENT_CACHE[cache_key] = result
        return result


    # -------------------------------
    # 3. TTS 요청
    # -------------------------------
    if is_tts:
        primary_id = None
        tts_candidates: List[Dict] = []

        if direct_title_matches:
            primary_id = direct_title_matches[0]["id"]
            tts_candidates = direct_title_matches
            reason = f"'{title_kw}'라는 작품 제목과 일치하는 작품을 찾아 음성 설명을 재생합니다."

        elif direct_artist_matches:
            primary_id = direct_artist_matches[0]["id"]
            tts_candidates = direct_artist_matches
            reason = f"'{artist_kw}' 작가의 작품 중 하나를 선택해 음성 설명을 재생합니다."

        # 🆕 vision_caption_ko 기반 매칭도 TTS에 사용
        elif caption_matches:
            primary_id = caption_matches[0]["id"]
            tts_candidates = caption_matches
            reason = f"'{subject_kw}'이(가) 포함된 장면이 Vision 캡션에 있는 작품을 선택해 음성 설명을 재생합니다."

        elif candidates:
            primary_id = candidates[0]["id"]
            tts_candidates = candidates
            reason = "의미검색 결과 중 가장 관련성이 높은 작품으로 음성 설명을 재생합니다."

        else:
            primary_id = pick_random_id(fallback_cat)
            tts_candidates = []
            reason = "검색 결과가 없어 임의의 작품으로 음성 설명을 재생합니다."

        print("[agent_route] forced tts:", primary_id)

        result = {
            "action": "tts",
            "primary_id": primary_id,
            "secondary_id": None,
            "category": fallback_cat,
            "reason": reason,
            "candidates": tts_candidates,
        }
        AGENT_CACHE[f"{fallback_cat}::{q}"] = result  # 캐시도 쓰고 있다면 이렇게
        return result


    # -------------------------------
    # 4. 후보가 하나도 없으면 랜덤 curate
    # -------------------------------
    if not candidates:
        rnd_id = pick_random_id(fallback_cat)
        result = {
            "action": "curate",
            "primary_id": rnd_id,
            "secondary_id": None,
            "category": fallback_cat,
            "reason": "의미검색 결과가 없어, 임의의 작품을 추천했습니다.",
            "candidates": [],
        }
        AGENT_CACHE[cache_key] = result
        return result

    # -------------------------------
    # 5. 일반 케이스: LLM에게 action 선택을 맡김
    # -------------------------------
    prompt_lines = [
        "당신은 미술관 AI 서비스의 '라우팅 에이전트'입니다.",
        "사용자의 한 줄 요청을 보고, 아래 세 가지 중 어떤 기능으로 보내면 좋을지 결정하세요.",
        "",
        "1) 'curate': 특정 작품 하나에 대한 큐레이터 설명을 보여주는 화면 (상세 화면).",
        "2) 'compare': 두 작품을 나란히 비교해 주는 화면.",
        "3) 'tts': 작품 하나의 설명을 들려주는 TTS 중심 화면 (라우팅은 상세 화면과 동일).",
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
        "- 사용자가 '읽어줘', '설명 들어보고 싶어', '음성', 'tts', '들려줘' 등을 언급하면 action은 'tts'를 사용하세요.",
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
        print("[agent_route] LLM raw:", raw)
    except Exception as e:
        print("[agent_route] LLM error:", e)
        if candidates:
            result = {
                "action": "curate",
                "primary_id": candidates[0]["id"],
                "secondary_id": None,
                "category": fallback_cat,
                "reason": "에이전트 모델 호출에 실패해서 첫 번째 후보를 기본 추천으로 사용했습니다.",
                "candidates": candidates,
            }
        else:
            result = {
                "action": "curate",
                "primary_id": pick_random_id(fallback_cat),
                "secondary_id": None,
                "category": fallback_cat,
                "reason": "에이전트 모델 호출에 실패해서 임의의 작품을 추천했습니다.",
                "candidates": [],
            }

        AGENT_CACHE[cache_key] = result
        return result

    try:
        parsed = json.loads(raw)
    except Exception:
        print("[agent_route] JSON parse failed, fallback to first candidate")
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

    result = {
        "action": action,
        "primary_id": primary_id,
        "secondary_id": secondary_id,
        "category": category,
        "reason": reason,
        "candidates": candidates,
    }
    AGENT_CACHE[cache_key] = result
    return result


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

    # 프론트에서 넘어온 값 정리
    language_code = (req.language_code or "ko-KR").strip()
    voice_name = (req.voice_name or "").strip() or "ko-KR-Wavenet-A"

    # 백엔드에서는 기본 1.0으로 두고, 실제 배속은 브라우저 audio.playbackRate로 제어
    try:
        speaking_rate = float(req.speaking_rate or 1.0)
    except (TypeError, ValueError):
        speaking_rate = 1.0

    # 너무 극단적인 값 방지 (선택사항)
    if speaking_rate < 0.7:
        speaking_rate = 0.7
    if speaking_rate > 2.0:
        speaking_rate = 2.0

    # 🔑 캐시 키: (언어, 보이스, 속도, 텍스트 md5)
    text_hash = md5(text.encode("utf-8")).hexdigest()
    cache_key = f"{language_code}|{voice_name}|{speaking_rate:.2f}|{text_hash}"

    # 1) 캐시 먼저 확인
    if cache_key in TTS_CACHE:
        return {"audio_b64": TTS_CACHE[cache_key]}

    # 2) Google TTS 호출
    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        audio_b64 = base64.b64encode(response.audio_content).decode("utf-8")

        # 3) 캐시에 저장
        TTS_CACHE[cache_key] = audio_b64

        return {"audio_b64": audio_b64}

    except Exception as e:
        print("[/ai/tts] error:", e)
        raise HTTPException(status_code=500, detail=f"TTS 실패: {e}")


# ───────────────────────────────────────────────────────────
# 유사한 이미지 엔드포인트
# ───────────────────────────────────────────────────────────    
@app.get("/similar_images")
def similar_images(
    id: str,
    category: Optional[str] = None,
    k: int = 6,
):
    """
    프론트에서 요청하는 유사 작품 추천 API

    - 성공: {"items": [ ... ]}
    - 실패/에러: {"items": []}   ← 500 안 던지고 그냥 빈 배열
    """
    # 이미지 검색 기능이 꺼져 있으면 바로 빈 리스트
    if not use_image_retriever or image_collection is None:
        print("[/similar_images] image retriever disabled")
        return {"items": []}

    try:
        items = similar_images_by_id(
            base_id=id,
            k=k,
            category=category,
        )
        # 항상 items 키로 리턴 (프론트 Detail.jsx와 맞추기)
        return {"items": items}
    except Exception as e:
        print(f"[/similar_images] error for id={id}: {e}")
        return {"items": []}

@app.get("/db_ids")
def db_ids():
    if image_collection is None:
        return {"ids": [], "note": "image_collection is None"}

    try:
        res = image_collection.get()
        ids = res.get("ids", [])
        # numpy array 방어
        if not isinstance(ids, list):
            ids = list(ids)
        return {"ids": ids}
    except Exception as e:
        return {"error": str(e), "ids": []}
