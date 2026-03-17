import os, json, glob
import chromadb
from chromadb import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing in environment")

genai.configure(api_key=API_KEY)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# ── Gemini 임베딩 함수 ─────────────────────────────────────
class GeminiEF(EmbeddingFunction):
    def __call__(self, texts):
        # texts: List[str] → List[List[float]]
        out = []
        for t in texts:
            r = genai.embed_content(model=EMBED_MODEL, content=t)
            out.append(r["embedding"])
        return out

# ── 벡터DB 초기화 (./chroma_db 에 영구 저장) ───────────────
client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(
    name="curator_corpus",
    embedding_function=GeminiEF(),
    metadata={"hnsw:space": "cosine"},
)

def load_cards_from_folder(folder):
    """AI-Hub 등 JSON 메타를 폴더에서 모아오기. 파일 구조에 맞게 수정."""
    items = []
    for path in glob.glob(os.path.join(folder, "**", "*.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # 예시: 파일 구조가 제각각일 수 있으니 필드명은 상황에 맞게 매핑
        # 아래는 안전하게 "가능하면 뽑기" 방식
        title = (data.get("Description",{}).get("ArtTitle_kor")
                 or data.get("Description",{}).get("ArtTitle_eng")
                 or data.get("Object_Info",{}).get("Title")
                 or os.path.basename(path))
        artist = (data.get("Description",{}).get("Artist")
                  or data.get("Description",{}).get("Artist_kor")
                  or data.get("Description",{}).get("Artist_eng")
                  or "")
        klass = (data.get("Description",{}).get("Class_kor")
                 or data.get("Object_Info",{}).get("MainCategory",""))
        mat = (data.get("Image_Info",{}).get("Material")
               or data.get("Description",{}).get("Material")
               or "")
        year = (data.get("Photo_Info",{}).get("PhotoDate")
                or data.get("Description",{}).get("Date")
                or "")

        # 검색/생성에 쓸 텍스트 본문(컨텍스트). 필요시 더 붙이거나 정제
        body_parts = [
            f"제목: {title}",
            f"작가: {artist}",
            f"분류: {klass}",
            f"재질: {mat}",
            f"연도/시기: {year}",
            # f"json_path: {path}",
        ]
        body = "\n".join([p for p in body_parts if p and p])

        items.append({
            "id": path.replace("\\", "/"),
            "text": body,
            "metadata": {
                "title": title, "artist": artist, "class": klass,
                "material": mat, "year": year, "json_path": path
            }
        })
    return items

if __name__ == "__main__":
    # ✅ 여기를 데이터 경로에 맞게 수정
    DATA_ROOTS = [
        r"D:\Exhibit\image_extracted",     # 예시
        r"D:\Exhibit\json_extracted",      # 예시
    ]

    docs, ids, metas = [], [], []
    for root in DATA_ROOTS:
        if not os.path.exists(root): 
            continue
        items = load_cards_from_folder(root)
        for it in items:
            ids.append(it["id"])
            docs.append(it["text"])
            metas.append(it["metadata"])

    if not ids:
        print("No JSON found. Adjust DATA_ROOTS or loader.")
        raise SystemExit(0)

    # 중복 방지: 이미 있는 id는 삭제 후 재추가(간단한 재색인 전략)
    try:
        existing = set(collection.get(ids=ids)["ids"])
    except Exception:
        existing = set()
    new_ids, new_docs, new_metas = [], [], []
    for i, d, m in zip(ids, docs, metas):
        if i in existing:
            collection.delete(ids=[i])
        new_ids.append(i); new_docs.append(d); new_metas.append(m)

    print(f"Adding {len(new_ids)} docs...")
    collection.add(ids=new_ids, documents=new_docs, metadatas=new_metas)
    print("Index built ✅  (./chroma_db)")
