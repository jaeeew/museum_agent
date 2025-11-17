import os
import json
import pandas as pd
from pathlib import Path

# ✅ 0. 기본 경로 (네가 보여준 구조 기준)
BASE_DIR = Path(r"E:\207.디지털 K-Art 데이터\01-1.정식개방데이터")

# 라벨(JSON) 폴더들
LABEL_DIRS = [
    BASE_DIR / r"Training\02.라벨링데이터",
    BASE_DIR / r"Validation\02.라벨링데이터",
]

# 이미지(원천데이터) 폴더들
IMG_DIRS = [
    BASE_DIR / r"Training\01.원천데이터",
    BASE_DIR / r"Validation\01.원천데이터",
]


# ✅ 1. 전체 이미지 파일을 미리 인덱싱 (이름 -> 전체 경로)
def index_images(img_dirs):
    index = {}
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    for root in img_dirs:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts):
                    stem = Path(f).stem.lower()  # 확장자 뺀 파일명
                    full_path = Path(dirpath) / f
                    # 같은 이름이 있으면 처음 것만 사용
                    index.setdefault(stem, full_path)

    print(f"✅ 인덱싱된 이미지 개수: {len(index)}")
    return index


# ✅ 2. 라벨(JSON) 모두 읽어서 메타데이터 생성
def collect_metadata(label_dirs, img_index):
    records = []

    for label_root in label_dirs:
        if not label_root.exists():
            continue

        # Training / Validation 구분
        split = "Training" if "Training" in str(label_root) else "Validation"

        for dirpath, dirnames, filenames in os.walk(label_root):
            for f in filenames:
                if not f.lower().endswith(".json"):
                    continue

                json_path = Path(dirpath) / f

                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                except Exception as e:
                    print(f"❌ JSON 읽기 실패: {json_path} -> {e}")
                    continue

                # ====== JSON에서 필요한 정보 뽑기 (네가 올린 구조 기준) ======
                obj_info = data.get("Object_Info", {})
                photo_info = data.get("Photo_Info", {})
                data_info = data.get("Data_Info", {})
                desc_info = data.get("Description", {})

                image_stem = data_info.get("ImageFileName")  # kart_2d000496-C-8-81-1
                ext = data_info.get("SourceDataExtension", "JPG").lower()

                img_path = None
                if image_stem:
                    key = image_stem.lower()
                    # 인덱스에서 찾기 (확장자는 상관없이 stem으로 매칭)
                    img_path = img_index.get(key)

                record = {
                    "split": split,  # Training / Validation
                    "json_path": str(json_path),

                    "image_id": image_stem,
                    "img_path": str(img_path) if img_path else None,

                    "main_category": obj_info.get("MainCategory"),
                    "sub_category": obj_info.get("SubCategory"),
                    "middle_category": obj_info.get("MiddleCategory"),

                    "class_kor": desc_info.get("Class_kor"),
                    "class_eng": desc_info.get("Class_eng"),

                    "title_kor": desc_info.get("ArtTitle_kor"),
                    "title_eng": desc_info.get("ArtTitle_eng"),

                    "artist_kor": desc_info.get("ArtistName_kor"),
                    "artist_eng": desc_info.get("ArtistName_eng"),

                    "location_kor": desc_info.get("Location_kor"),
                    "location_eng": desc_info.get("Location_eng"),

                    "material_kor": desc_info.get("Material_kor"),
                    "material_eng": desc_info.get("Material_eng"),

                    "photo_date": photo_info.get("PhotoDate"),
                    "photo_equipment": photo_info.get("PhotoEquipment"),

                    "data_sort": data_info.get("DataSort"),
                    "source_ext": data_info.get("SourceDataExtension"),
                    "license": data_info.get("Rangeofuse"),
                }

                records.append(record)

    return records


def main():
    # 1) 이미지 인덱싱
    img_index = index_images(IMG_DIRS)

    # 2) JSON -> 메타데이터 목록 만들기
    records = collect_metadata(LABEL_DIRS, img_index)

    print(f"✅ 메타데이터 레코드 수: {len(records)}")

    # 3) CSV 저장
    out_path = BASE_DIR / "k_art_metadata.csv"
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"🎉 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
