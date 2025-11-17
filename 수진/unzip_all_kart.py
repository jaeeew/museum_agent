import os
import zipfile

# ✅ 1. 기준 폴더 (네 스샷 기준)
BASE_DIR = r"E:\207.디지털 K-Art 데이터\01-1.정식개방데이터"

# ✅ 2. 압축 파일들이 들어 있는 하위 폴더들
SUB_DIRS = [
    r"Training\01.원천데이터",
    r"Training\02.라벨링데이터",
    r"Validation\01.원천데이터",
    r"Validation\02.라벨링데이터",
]

def unzip_in_folder(folder_path: str):
    """해당 폴더 안의 모든 .zip 파일을 각각의 폴더에 풀어줌"""
    print(f"\n📂 폴더 검사 중: {folder_path}")

    if not os.path.isdir(folder_path):
        print(f"  ⚠️ 폴더가 없음: {folder_path}")
        return

    for name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, name)

        # 파일이 아니면 패스 (혹시 폴더가 섞여 있을 수도 있으니까)
        if not os.path.isfile(file_path):
            continue

        # 확장자가 .zip 인 것만 처리
        if not name.lower().endswith(".zip"):
            continue

        # 압축 풀 위치: 같은 폴더 안에 "파일이름_폴더"
        zip_name_no_ext = os.path.splitext(name)[0]
        extract_to = os.path.join(folder_path, zip_name_no_ext)

        print(f"\n=== 🗜 압축 해제 시작: {name}")
        print(f"    ➜ {extract_to}")

        os.makedirs(extract_to, exist_ok=True)

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(extract_to)
        except Exception as e:
            print(f"    ❌ 압축 해제 실패: {e}")
        else:
            print(f"    ✅ 압축 해제 완료")

def main():
    for sub in SUB_DIRS:
        folder = os.path.join(BASE_DIR, sub)
        unzip_in_folder(folder)

    print("\n🎉 모든 작업이 끝났습니다!")

if __name__ == "__main__":
    main()
