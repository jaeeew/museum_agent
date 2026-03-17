# make_cards_fast.py  (방법 B: 공개용 별도 출력 + 웹 번들 자동 생성)
from pathlib import Path
import json

# ───────────────────────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────────────────────
IMG_ROOT   = Path(r"D:\Exhibit\image_extracted")
JSON_ROOT  = Path(r"D:\Exhibit\json_extracted")
OUT_DIR    = Path(r"D:\Exhibit\export")
WEB_DIR    = Path(r"D:\Exhibit\web")                    # 프론트 정적 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

CARDS_PATH         = OUT_DIR / "cards.jsonl"            # 내부 원본(민감 포함 가능)
PUBLIC_CARDS_PATH  = OUT_DIR / "cards_public.jsonl"     # 공개용(민감 제거)
INDEX_PATH         = OUT_DIR / "search_index.jsonl"     # 검색 인덱스
WEB_BUNDLE_PATH    = WEB_DIR / "cards.min.json"         # 프론트에서 읽는 배열 JSON

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ───────────────────────────────────────────────────────────
# 공개/민감 필드 정책
# ───────────────────────────────────────────────────────────
PUBLIC_KEYS = {
    "id","title","title_ko","title_en",
    "artist","artist_ko","artist_en",
    "class","class_ko","class_en",
    "categories","material","material_ko","material_en",
    "dimensions","date_or_period","photo_date","tags","image_path"
}
SENSITIVE_KEYS = {"rights", "json_path", "source_ext", "license", "usage", "usage_rights"}

def to_public(card: dict) -> dict:
    """공개용으로 내보낼 때 허용 키만 남기고, 방어적으로 민감 키 제거"""
    pub = {k: v for k, v in card.items() if k in PUBLIC_KEYS}
    for k in SENSITIVE_KEYS:
        pub.pop(k, None)
    return pub

# ───────────────────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────────────────
def read_json_any(p: Path):
    for enc in ("utf-8","utf-8-sig","cp949","euc-kr"):
        try:
            return json.loads(p.read_text(encoding=enc))
        except Exception:
            pass
    raise RuntimeError(f"JSON decode failed: {p}")

def first(*vals):
    for v in vals:
        if v not in (None, "", "N/A"):
            return v
    return ""

def get(d, path, default=""):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default

def build_image_index():
    print("[1/4] 이미지 인덱스 생성 중…")
    idx = {}
    cnt = 0
    for p in IMG_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.stem.lower(), str(p))
            cnt += 1
            if cnt % 5000 == 0:
                print(f"  - 인덱싱 {cnt}개…")
    print(f"  - 완료: {cnt}개 파일 / 고유 stem {len(idx)}개")
    return idx

# ───────────────────────────────────────────────────────────
# 카드/인덱스 생성
# ───────────────────────────────────────────────────────────
def record_to_card(data: dict, json_path: Path, img_idx: dict):
    stem = first(get(data,"Data_Info.ImageFileName"), json_path.stem)
    title_ko = get(data,"Description.ArtTitle_kor")
    title_en = get(data,"Description.ArtTitle_eng")
    artist_ko = first(get(data,"Description.ArtistName_kor"), get(data,"Description.Artist_kor"))
    artist_en = first(get(data,"Description.ArtistName_eng"), get(data,"Description.Artist_eng"))
    class_ko = get(data,"Description.Class_kor"); class_en = get(data,"Description.Class_eng")
    main_cat = get(data,"Object_Info.MainCategory"); sub_cat = get(data,"Object_Info.SubCategory"); mid_cat = get(data,"Object_Info.MiddleCategory")
    material_ko = get(data,"Description.Material_kor"); material_en = get(data,"Description.Material_eng")
    width  = get(data,"Image_Info.Width"); length = get(data,"Image_Info.Length"); height = get(data,"Image_Info.Height")
    rights = get(data,"Data_Info.Rangeofuse")  # 내부 원본에는 유지해도 됨(공개 변환에서 제거)
    source_ext = (get(data,"Data_Info.SourceDataExtension") or "jpg").lower()
    photo_date = get(data,"Photo_Info.PhotoDate")

    image_path = img_idx.get(stem.lower(), "")

    tags = [t for t in [class_ko, main_cat, sub_cat, mid_cat] if t]
    return {
        "id": stem,
        "title": first(title_ko, title_en),
        "title_ko": title_ko,
        "title_en": title_en,
        "artist": first(artist_ko, artist_en),
        "artist_ko": artist_ko,
        "artist_en": artist_en,
        "class": first(class_ko, class_en),
        "class_ko": class_ko,
        "class_en": class_en,
        "categories": [c for c in [main_cat, sub_cat, mid_cat] if c],
        "material": first(material_ko, material_en),
        "material_ko": material_ko,
        "material_en": material_en,
        "dimensions": {"width": width, "length": length, "height": height},
        "date_or_period": photo_date,
        "photo_date": photo_date,
        "rights": rights,                 # 내부 원본용
        "source_ext": source_ext,         # 내부 원본용
        "image_path": image_path,
        "json_path": str(json_path),      # 내부 원본용
        "tags": tags
    }

def card_to_search_doc(card: dict):
    # ✅ 검색 인덱스는 rights 제거
    pieces = [
        card.get("title_ko") or card.get("title_en") or "",
        card.get("artist_ko") or card.get("artist_en") or "",
        card.get("class_ko") or card.get("class_en") or "",
        " ".join(card.get("categories", [])),
        card.get("material_ko") or card.get("material_en") or "",
        card.get("date_or_period") or "",
        # card.get("rights") or "",   # 포함 금지
    ]
    plain = " | ".join([p for p in pieces if p])
    return {
        "id": card["id"],
        "title": card["title"],
        "artist": card["artist"],
        "plain_text": plain,
        "tags": card.get("tags", []),
        "image_path": card.get("image_path","")
    }

# ───────────────────────────────────────────────────────────
# 공개용 번들(cards.min.json) 생성
# ───────────────────────────────────────────────────────────
def write_web_bundle_from_public_jsonl(src_jsonl: Path, out_json: Path):
    items = []
    cnt = 0
    with src_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 방어적으로 한 번 더 민감키 제거
            for k in SENSITIVE_KEYS:
                obj.pop(k, None)
            items.append(obj)
            cnt += 1
    out_json.write_text(json.dumps(items, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[4/4] 웹 번들 생성 완료 → {out_json}  (items={cnt})")

# ───────────────────────────────────────────────────────────
# 메인
# ───────────────────────────────────────────────────────────
def main():
    img_idx = build_image_index()

    json_files = list(JSON_ROOT.rglob("*.json"))
    total = len(json_files)
    print(f"[2/4] JSON 처리 시작… 총 {total}개")

    # 매번 깔끔하게 재생성(덮어쓰기). append 방식 원하면 "a"로 바꾸세요.
    with CARDS_PATH.open("w", encoding="utf-8") as cf, \
         PUBLIC_CARDS_PATH.open("w", encoding="utf-8") as pcf, \
         INDEX_PATH.open("w", encoding="utf-8") as inf:

        for i, jf in enumerate(json_files, 1):
            try:
                data = read_json_any(jf)
                card = record_to_card(data, jf, img_idx)
                idx  = card_to_search_doc(card)

                # 내부 원본
                cf.write(json.dumps(card, ensure_ascii=False) + "\n")
                # 공개용
                pcf.write(json.dumps(to_public(card), ensure_ascii=False) + "\n")
                # 검색 인덱스
                inf.write(json.dumps(idx, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"  ! 오류: {jf} | {e}")

            if i % 500 == 0:
                print(f"  - {i}/{total} 진행중…")

    print(f"[3/4] 완료: cards={CARDS_PATH}  public={PUBLIC_CARDS_PATH}  index={INDEX_PATH}")

    # 공개용 jsonl → 웹 번들 배열 JSON
    write_web_bundle_from_public_jsonl(PUBLIC_CARDS_PATH, WEB_BUNDLE_PATH)

if __name__ == "__main__":
    main()
