import os
import json
import random
from typing import Optional, Dict, List
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

import base64

# ── Google Gemini ─────────────────────────────────
import google.generativeai as genai

# ── Google Cloud TTS (옵션) ───────────────────────
from google.cloud import texttospeech

# ── Chroma + Gemini 임베딩 ───────────────────────
import chromadb
from chromadb import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

# ── CLIP (텍스트/이미지 임베딩) ───────────────────
import torch
import open_clip  # pip install open_clip_torch
from PIL import Image


# ─────────────────────────────────────────────────
# 환경설정
# ─────────────────────────────────────────────────
load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-004")

if API_KEY:
    genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(MODEL_NAME)


# ─────────────────────────────────────────────────
# 데이터 루트 (작품 JSON/이미지 경로)
#   예: E:\207.디지털 K-Art 데이터\01-1.정식개방데이터
# ─────────────────────────────────────────────────
DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        r"E:\207.디지털 K-Art 데이터\01-1.정식개방데이터",
    )
)

# JSON은 Training/02.라벨링데이터 아래 TL_01... 폴더
JSON_ROOT = DATA_ROOT / "Training" / "02.라벨링데이터"

# 이미지는 Training + Validation 통째로 검색
IMG_ROOT = DATA_ROOT

IMAGE_EXTS = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")


@lru_cache(maxsize=1)
def build_image_index() -> Dict[str, str]:
    """
    IMG_ROOT 전체를 한 번만 스캔해서
    파일이름(stem) → 상대경로 를 캐싱해 둔다.
    """
    index: Dict[str, str] = {}
    print("[IMAGE_INDEX] building index...")
    for ext in IMAGE_EXTS:
        for path in IMG_ROOT.rglob(f"*.{ext}"):
            stem = path.stem  # kart_2d000645-C-8-81-1
            rel = path.relative_to(IMG_ROOT).as_posix()
            if stem not in index:
                index[stem] = rel
    print(f"[IMAGE_INDEX] built index for {len(index)} images")
    return index


def find_image_path_for_prefix(prefix: str) -> Optional[Path]:
    """
    build_image_index()를 사용해서 prefix에 해당하는 실제 이미지 Path 반환
    """
    index = build_image_index()

    # 1) 완전 일치
    rel = index.get(prefix)
    if rel:
        return IMG_ROOT / rel

    # 2) prefix로 시작하는 것
    for k, v in index.items():
        if k.startswith(prefix):
            return IMG_ROOT / v

    return None


# ─────────────────────────────────────────────────
# 카테고리 별명 → 실제 폴더명 매핑
# ─────────────────────────────────────────────────
CATEGORY_MAP: Dict[str, str] = {
    "painting_json": "TL_01. 2D_02.회화(Json)",
    "craft_json": "TL_01. 2D_04.공예(Json)",
    "sculpture_json": "TL_01. 2D_06.조각(Json)",
}


def map_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    return CATEGORY_MAP.get(cat, cat)


# ─────────────────────────────────────────────────
# Chroma (텍스트 RAG + Gemini 임베딩)
# ─────────────────────────────────────────────────
use_retriever = True
use_image_retriever = True

retrieval = None
image_collection = None


class GeminiEF(EmbeddingFunction):
    def __call__(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            r = genai.embed_content(model=EMBED_MODEL, content=t)
            out.append(r["embedding"])
        return out


try:
    # 텍스트 RAG
    text_client = chromadb.PersistentClient(
        path="./chroma_text",
        settings=Settings(anonymized_telemetry=False),
    )
    retrieval = text_client.get_or_create_collection(
        name="curator_corpus",
        embedding_function=GeminiEF(),
        metadata={"hnsw:space": "cosine"},
    )

    # 이미지 RAG (CLIP 임베딩이 이미 들어있다고 가정)
    image_client = chromadb.PersistentClient(
        path="./chroma_image",
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


# ─────────────────────────────────────────────────
# CLIP 모델 (텍스트→이미지 검색 / 이미지 분석)
# ─────────────────────────────────────────────────
try:
    CLIP_MODEL_NAME = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    clip_model = clip_model.to(clip_device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    @torch.no_grad()
    def embed_clip_text(texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        tokens = clip_tokenizer(texts).to(clip_device)
        feats = clip_model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().tolist()

    @torch.no_grad()
    def embed_clip_image(img_path: Path) -> Optional[List[float]]:
        if not img_path.exists():
            return None
        img = Image.open(img_path).convert("RGB")
        image_tensor = clip_preprocess(img).unsqueeze(0).to(clip_device)
        feats = clip_model.encode_image(image_tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().tolist()

except Exception as e:
    print("[WARN] CLIP 초기화 실패:", e)
    use_image_retriever = False
    clip_model = None
    clip_tokenizer = None
    clip_preprocess = None


# ─────────────────────────────────────────────────
# FastAPI 앱 & CORS
# ─────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지
app.mount(
    "/image_extracted",
    StaticFiles(directory=str(IMG_ROOT)),
    name="images",
)

# JSON
app.mount(
    "/json_extracted",
    StaticFiles(directory=str(JSON_ROOT)),
    name="json",
)


# ─────────────────────────────────────────────────
# Pydantic 모델
# ─────────────────────────────────────────────────
class CurateIn(BaseModel):
    id: str
    card: Optional[Dict] = None


class CurateImmersiveIn(BaseModel):
    id: Optional[str] = None
    category: Optional[str] = None
    card: Optional[Dict] = None


class CompareIn(BaseModel):
    ids: List[str]
    category: Optional[str] = None
    locale: Optional[str] = "ko"


class AgentIn(BaseModel):
    query: str
    category: Optional[str] = None
    locale: Optional[str] = "ko"


class TtsIn(BaseModel):
    text: str
    language_code: Optional[str] = "ko-KR"
    voice_name: Optional[str] = None
    speaking_rate: Optional[float] = 1.0


# ─────────────────────────────────────────────────
# 유틸: 카드 로딩
# ─────────────────────────────────────────────────
def load_card_by_id(category: Optional[str], art_id: str) -> Dict:
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


# ─────────────────────────────────────────────────
# RAG 유틸
# ─────────────────────────────────────────────────
def build_query(card: Dict) -> str:
    desc = card.get("Description") or {}
    photo = card.get("Photo_Info") or {}
    data_info = card.get("Data_Info") or {}

    parts: List[Optional[str]] = [
        card.get("title")
        or desc.get("ArtTitle_kor")
        or desc.get("ArtTitle_eng")
        or data_info.get("ImageFileName"),
        card.get("artist")
        or desc.get("ArtistName_kor")
        or desc.get("ArtistName_eng"),
        card.get("class")
        or desc.get("Class_kor")
        or desc.get("Class_eng"),
        card.get("material")
        or desc.get("Material_kor")
        or desc.get("Material_eng"),
        card.get("date_or_period") or photo.get("PhotoDate"),
    ]
    return " ".join([p for p in parts if p])


def retrieve_context(query: str, k: int = 5) -> List[Dict]:
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
        hits.append(
            {
                "id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "meta": metas[i] if i < len(metas) else {},
                "score": dists[i] if i < len(dists) else None,
            }
        )
    return hits


def format_context(hits: List[Dict]) -> str:
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
    if not use_image_retriever or not image_collection or not query:
        return []

    try:
        vec = embed_clip_text([query])[0]
    except Exception as e:
        print("[WARN] CLIP embed 실패:", e)
        return []

    res = image_collection.query(query_embeddings=[vec], n_results=k)
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


# ─────────────────────────────────────────────────
# CLIP 기반 라벨링
# ─────────────────────────────────────────────────
VISUAL_LABEL_CANDIDATES: Dict[str, List[str]] = {
    "새": ["새", "작은 새 두 마리", "새 한 마리", "새가 앉아 있는 모습"],
    "꽃": ["꽃", "매화 꽃", "벚꽃 가지", "꽃송이"],
    "나뭇가지": ["나뭇가지", "가는 가지", "나무 가지", "매화 가지"],
    "나무": ["나무", "큰 나무", "고목", "소나무"],
    # 물·연못 계열
    "물": ["물", "연못", "물결", "물가", "물 위의 반사"],
    # 물고기/금붕어
    "물고기": ["물고기", "금붕어", "연못 속 물고기", "헤엄치는 물고기"],
    # 인물/동물/과일/글씨
    "사람": ["사람", "인물 한 명", "인물 두 명", "선비 한 명"],
    "다람쥐": ["다람쥐", "작은 다람쥐", "나뭇가지 위의 다람쥐"],
    "과일": ["과일", "감", "귤", "복숭아", "포도", "나무에 달린 과일"],
    "글씨": ["글씨", "서예", "먹으로 쓴 글씨", "한자 글씨"],
}

def _cos_sim(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def analyze_image_labels(img_path: Path, top_k: int = 4) -> List[str]:
    """
    CLIP으로 이미지와 텍스트 후보를 비교해서
    그 그림에 '있어 보이는' 대상 라벨을 상위 top_k개 뽑아줌.
    """
    if clip_model is None or embed_clip_image is None:
        return []

    try:
        img_vec = embed_clip_image(img_path)
    except Exception as e:
        print("[WARN] CLIP 이미지 분석 실패:", e)
        return []

    if not img_vec:
        return []

    labels_with_scores: List[tuple[str, float]] = []

    for label, phrases in VISUAL_LABEL_CANDIDATES.items():
        try:
            text_vecs = embed_clip_text(phrases)
        except Exception as e:
            print("[WARN] CLIP 텍스트 임베딩 실패:", e)
            continue

        best = max(_cos_sim(img_vec, tv) for tv in text_vecs)
        labels_with_scores.append((label, float(best)))

    labels_with_scores.sort(key=lambda x: x[1], reverse=True)

    # 너무 애매한(점수 낮은) 라벨은 버림
    filtered = [lb for lb, sc in labels_with_scores if sc > 0.27]
    return filtered[:top_k]


# ─────────────────────────────────────────────────
# 프롬프트 빌더 (일반 텍스트 모드)
# ─────────────────────────────────────────────────
def build_prompt(card: Dict, context_block: str) -> str:
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

    year = card.get("date_or_period") or photo.get("PhotoDate") or ""

    cats = " / ".join(card.get("categories", []) or [])
    lines = [
        "당신은 국공립 미술관의 전문 큐레이터입니다. 차분하고 따뜻한 말투로, 관람객에게 편안히 이야기하듯 한국어 구어체로 설명하세요.",
        "설명은 3~4개의 짧은 단락, 총 5~7문장으로 작성합니다. 제목/번호/불릿/이모지/괄호 표시는 사용하지 마세요.",
        "이 작품이 말하는 ‘핵심 의미/주제’를 서두 1~2문장에서 선명하게 제시하고, 나머지 정보는 배경 수준으로만 간결히 덧붙이세요.",
        "형식·재료 분석은 핵심 1~2포인트만 짧게 언급하고, 감상 포인트도 1~2문장으로 권유형 종결을 사용하세요.",
        "권리·이용범위·라이선스·파일 경로·출처 표기는 언급하지 마세요.",
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


# ─────────────────────────────────────────────────
# 프롬프트 빌더 (몰입형/Immersive 전용)
# ─────────────────────────────────────────────────
def build_immersive_prompt(
    card: Dict,
    context_block: str,
    visual_labels: List[str],
) -> str:
    """
    몰입형(TTS + 화면 이동) 전용 해설 프롬프트.
    - 1문단: 작품 전체 분위기/의미 (2~3문장, 방향 표현 금지)
    - 2문단: 화면 왼쪽/가운데/오른쪽을 순서대로 설명 (각각 한 문장 이상)
    - 3문단: 감정·메시지 정리 (2~3문장, 방향 표현 금지)
    """
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

    # 제목/분류/주제 같은 메타데이터에서 단어를 조금 더 가져와서
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

    # CLIP 라벨 + 메타데이터 단어 합치고, 중복 제거
    allowed_targets_list = list(dict.fromkeys((visual_labels or []) + meta_targets))
    labels_str = (
        ", ".join(allowed_targets_list) if allowed_targets_list else "특별히 추출된 단어 없음"
    )

    _ = context_block  # 현재는 사용 안 함

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
        "   - 두 번째 문단: 화면 왼쪽, 화면 가운데, 화면 오른쪽을 이 순서대로 설명하는 문장을 각각 한 번 이상 포함합니다.",
        "   - 세 번째 문단: 관람자가 느끼면 좋을 감정·메시지·여운을 2~3문장으로 정리합니다.",
        "2. 첫 번째 문단과 세 번째 문단에서는 '화면 왼쪽', '화면 가운데', '화면 오른쪽', '왼쪽', '가운데', '오른쪽' 같은 방향 표현을 사용하지 마십시오.",
        "3. 두 번째 문단에서는 반드시 다음 표현을 이 순서대로 한 번씩만 사용합니다: '화면 왼쪽에는', '화면 가운데에는', '화면 오른쪽에는'.",
        "4. 한 문단 안에서는 줄바꿈을 하지 말고, 자연스럽게 한 문단으로 이어서 작성합니다.",
        "5. 각 문단이 끝날 때마다 한 줄을 완전히 비우고, 다음 문단을 새 줄에서 시작하십시오. (즉, 문단 사이에 빈 줄 한 줄을 넣으십시오.)",
        "6. 번호, 불릿, 큰따옴표는 출력에 포함하지 말고, 오직 세 개의 문단 텍스트만 출력하십시오.",
    ]

    if any("새" in t for t in allowed_targets_list):
        lines.append(
            "7. 목록에 '새'가 있다면, 적어도 한 문장에서는 새의 위치나 모습을 구체적으로 설명하십시오."
        )
    if any("물고기" in t or "금붕어" in t for t in allowed_targets_list):
        lines.append(
            "8. 목록에 '물고기'나 '금붕어'가 있다면, 적어도 한 문장에서는 물고기의 색감이나 움직임을 구체적으로 설명하십시오."
        )
    if any("사람" in t for t in allowed_targets_list):
        lines.append(
            "9. 목록에 '사람'이 있다면, 인물의 자세나 표정을 한 문장 이상에서 설명하십시오."
        )

    return "\n".join(lines)



# ─────────────────────────────────────────────────
# 랜덤 ID 유틸
# ─────────────────────────────────────────────────
def list_ids_for_category(category: str = "painting_json") -> List[str]:
    real_cat = map_category(category)
    target_dir = JSON_ROOT / real_cat
    if not target_dir.exists() or not target_dir.is_dir():
        return []
    ids: List[str] = [p.stem for p in target_dir.glob("*.json")]
    return ids


def pick_random_id(category: str = "painting_json") -> Optional[str]:
    ids = list_ids_for_category(category)
    if not ids:
        return None
    return random.choice(ids)


# ─────────────────────────────────────────────────
# 라우트
# ─────────────────────────────────────────────────
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
    real_cat = map_category(category)
    target_dir = JSON_ROOT / real_cat
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"category not found: {real_cat}")
    files = [p.name for p in target_dir.glob("*.json")]
    return files


@app.get("/find_image/{prefix}")
def find_image(prefix: str):
    """
    prefix(예: kart_2d000645-C-8-81-1)에 해당하는 이미지를
    미리 만들어 둔 인덱스에서 빠르게 찾는다.
    """
    index = build_image_index()

    rel = index.get(prefix)
    if rel:
        return {"url": f"/image_extracted/{rel}"}

    for k, v in index.items():
        if k.startswith(prefix):
            return {"url": f"/image_extracted/{v}"}

    raise HTTPException(
        status_code=404,
        detail=f"image not found for prefix={prefix}",
    )


@app.post("/curate")
async def curate(req: CurateIn):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")

    card = req.card or {}
    # 카드가 비어 있으면 id로 다시 로드
    if not card and req.id:
        card = load_card_by_id(None, req.id)

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


# -----------------------------
# immersive 해설 생성 (안정 버전, async)
# -----------------------------
import asyncio
from google.api_core.exceptions import GoogleAPICallError, RetryError

@app.post("/curate/immersive")
async def curate_immersive(req: CurateImmersiveIn):

    try:
        card = req.card or {}
        if not card and req.id:
            try:
                card = load_card_by_id(req.category, req.id)
            except Exception:
                return {
                    "curator_text": "작품 정보를 불러오지 못했습니다.",
                    "labels": []
                }

        # ------------------------------
        # 이미지 라벨링 (timeout 5초)
        # ------------------------------
        visual_labels = []
        if req.id:
            img_path = find_image_path_for_prefix(req.id)
            if img_path:
                async def _clip():
                    return analyze_image_labels(img_path)

                try:
                    visual_labels = await asyncio.wait_for(_clip(), timeout=5)
                except:
                    visual_labels = []

        # ------------------------------
        # 프롬프트 생성 (우리가 수정한 3문단용 함수)
        # ------------------------------
        prompt = build_immersive_prompt(card, "", visual_labels)

        # ------------------------------
        # Gemini 호출 (timeout 15초)
        # ------------------------------
        async def _gemini():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )

        try:
            resp = await asyncio.wait_for(_gemini(), timeout=15)
            text = (resp.text or "").strip()

        except asyncio.TimeoutError:
            print("❌ Gemini TIMEOUT")
            text = (
                "이 작품은 전체적으로 고요하면서도 생동감 있는 분위기를 담고 있습니다. "
                "화면을 천천히 훑어 보며 색감과 구도를 감상해 보세요.\n\n"
                "화면 왼쪽에는, 화면 가운데에는, 화면 오른쪽에는 각각 다른 요소들이 자리하고 있으니 "
                "차례로 시선을 옮겨 보시기 바랍니다.\n\n"
                "지금은 AI 해설이 완전히 생성되지 않았지만, 작품이 주는 여운과 감정을 천천히 느껴 보세요."
            )

        except Exception as e:
            print("❌ Gemini ERROR:", e)
            text = (
                "현재 AI 해설을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요.\n\n"
                "그동안에는 화면 전체를 천천히 살펴보며, 색감과 구도, 등장하는 대상들을 직접 감상해 보시길 권합니다."
            )

        # 정상 응답
        return {
            "curator_text": text,
            "labels": visual_labels
        }

    except Exception as e:
        # 마지막 보호막
        print("❌ IMMERSIVE UNKNOWN ERROR:", e)
        return {
            "curator_text": "해설을 불러오는 중 문제가 발생했습니다.",
            "labels": []
        }

    
# ───────────────────────────────────────────────────────────
# Google Cloud TTS 엔드포인트
# 프론트에서 /ai/tts 로 POST 하면 base64 mp3 를 돌려준다.
# ───────────────────────────────────────────────────────────
@app.post("/ai/tts")
async def ai_tts(req: TtsIn):
    """
    Google Cloud Text-to-Speech 를 이용해서
    req.text 를 mp3 로 합성하고 base64 문자열로 반환.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="TTS text is empty")

    # 기본값 세팅
    language_code = req.language_code or "ko-KR"
    speaking_rate = req.speaking_rate or 1.0

    try:
        # GOOGLE_APPLICATION_CREDENTIALS 환경변수에 지정된
        # 서비스 계정 키(gcp-tts.json)를 자동으로 사용한다.
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        # voice_name 이 오면 그 이름을 그대로 사용 (예: "ko-KR-Wavenet-A")
        # 안 오면 language_code + NEUTRAL 로 기본 생성
        if req.voice_name:
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=req.voice_name,
            )
        else:
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
            )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        # 바이너리를 base64 문자열로 인코딩
        audio_b64 = base64.b64encode(response.audio_content).decode("utf-8")

        return {"audio_b64": audio_b64}

    except Exception as e:
        print("[ERROR] TTS 실패:", e)
        raise HTTPException(
            status_code=500,
            detail=f"TTS generation failed: {e}",
        )


@app.get("/search_image")
def search_image(q: str, k: int = 5):
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


# ─────────────────────────────────────────────────
# 간단한 에이전트 엔드포인트 (Welcome 화면에서 사용)
# ─────────────────────────────────────────────────
@app.post("/ai/agent")
async def ai_agent(req: AgentIn):
    """
    아주 간단한 rule 기반:
    - '비교', '두 작품' 등이 들어있으면 compare 모드
    - '읽어줘', '들려줘', '음성' 등이 있으면 tts 모드
    - 그 외에는 curate 모드 (단일 작품 추천)
    """
    q = req.query.strip()
    category = req.category or "painting_json"

    if not q:
        q = "오늘 볼 만한 작품을 추천해줘"

    lower_q = q.lower()
    action = "curate"
    if "비교" in q or "두 작품" in q or "둘 다" in q:
        action = "compare"
    elif "읽어줘" in q or "들려줘" in q or "음성" in q:
        action = "tts"

    # 추천 id 하나 뽑기 (retrieval이 있으면 거기서, 없으면 랜덤)
    primary_id = None
    secondary_id = None

    if use_retriever and retrieval:
        hits = retrieve_context(q, k=3)
        if hits:
            primary_id = hits[0]["id"]
            if action == "compare" and len(hits) > 1:
                secondary_id = hits[1]["id"]

    if not primary_id:
        primary_id = pick_random_id(category)
    if action == "compare" and not secondary_id:
        secondary_id = pick_random_id(category)

    return {
        "action": action,
        "primary_id": primary_id,
        "secondary_id": secondary_id,
        "category": category,
    }


# ─────────────────────────────────────────────────
# 비교 해설 (Compare 페이지에서 사용)
# ─────────────────────────────────────────────────
@app.post("/ai/analyze-compare")
async def ai_analyze_compare(req: CompareIn):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY")

    if len(req.ids) < 2:
        raise HTTPException(status_code=400, detail="at least 2 ids required")

    cards: List[Dict] = []
    for i in req.ids[:2]:
        cards.append(load_card_by_id(req.category, i))

    def card_brief(c: Dict) -> str:
        desc = c.get("Description") or {}
        photo = c.get("Photo_Info") or {}
        data_info = c.get("Data_Info") or {}
        title = (
            c.get("title")
            or desc.get("ArtTitle_kor")
            or desc.get("ArtTitle_eng")
            or data_info.get("ImageFileName")
            or c.get("id", "")
        )
        artist = (
            c.get("artist")
            or desc.get("ArtistName_kor")
            or desc.get("ArtistName_eng")
            or ""
        )
        klass = (
            c.get("class")
            or desc.get("Class_kor")
            or desc.get("Class_eng")
            or ""
        )
        year = c.get("date_or_period") or photo.get("PhotoDate") or ""
        material = (
            c.get("material")
            or desc.get("Material_kor")
            or desc.get("Material_eng")
            or ""
        )
        return f"- 제목: {title}, 작가: {artist}, 분류/장르: {klass}, 재질: {material}, 연도/시기: {year}"

    prompt = "\n".join(
        [
            "당신은 미술관 도슨트입니다.",
            "아래 두 작품의 공통점과 차이점을 비교해서 설명해 주세요.",
            "전문 용어를 남발하지 말고, 관람객이 이해하기 쉬운 한국어 구어체로 작성하세요.",
            "",
            "[작품 A]",
            card_brief(cards[0]),
            "",
            "[작품 B]",
            card_brief(cards[1]),
            "",
            "1~2문장으로 두 작품의 공통된 분위기·주제를 설명하고,",
            "이어지는 4~5문장에서 구체적인 차이점(구도, 색감, 인물/대상, 감정 표현 등)을 정리해 주세요.",
            "불릿/번호/제목 없이 자연스러운 단락 텍스트만 출력하세요.",
        ]
    )

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compare generation failed: {e}")

    return {"compare_text": text}
