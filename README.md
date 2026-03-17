# 🏛️ Museum Agent

## ✨ 개요
**Museum Agent**는 멀티모달 RAG 기반 **AI 도슨트 시스템**입니다.

사용자는 전시 작품을 탐색하면서  
AI가 생성한 작품 해설을 듣거나, 유사 작품을 추천받고,  
두 작품을 비교 분석하는 **몰입형 디지털 전시 경험**을 할 수 있습니다.

기존 전시 서비스가 단순한 이미지와 텍스트 설명 중심이었다면  
Museum Agent는 **텍스트 RAG + 이미지 RAG + Vision 모델**을 결합하여  
사용자의 질문과 의도에 맞는 **지능형 전시 해설 서비스**를 제공합니다.

---

# 📂 데이터

프로젝트에서는 **AI Hub 디지털 K-Art 데이터셋**을 활용했습니다.

## 데이터 구성

### 이미지 데이터
- 약 **92,547장** 작품 이미지 (JPG)

### 메타데이터
- 작품 제목
- 작가
- 시대
- 소재
- 작품 설명

### 메타데이터 예시

```json
{
  "MainCategory": "현대",
  "SubCategory": "한국화",
  "Title": "Intention",
  "ArtistName": "김영재",
  "Material": "종이"
}'''


# 🚀 주요 기능

## 🎨 AI 작품 해설 생성 (Curate)

- 작품 메타데이터 기반 **RAG 해설 생성**
- Google **Gemini LLM** 활용
- 작품의 **시대적 맥락과 의미 설명 제공**

---

## 🔍 유사 작품 추천 (Image RAG)

- **CLIP 이미지 임베딩** 활용
- 작품 스타일 기반 **유사 작품 추천**

Artwork → Image Embedding → Vector Search → Similar Artwork



---

## 🖼️ 몰입형 전시 (Immersive Mode)

- 작품 **확대 및 이동 (Zoom & Pan)**
- **AI 음성 해설 제공**
- **Google Cloud TTS** 기반 도슨트 음성 생성

---

## ⚖️ 작품 비교 (Compare)

사용자가 선택한 두 작품을 비교 분석

- 작품 **스타일**
- **소재**
- **시대적 맥락**

AI가 두 작품의 **차이점과 특징을 설명**합니다.

---

## 🤖 AI 라우팅 에이전트

사용자의 질문을 분석하여 기능을 자동 선택합니다.

### 예시

| 사용자 질문 | 실행 기능 |
|---|---|
| 이 작품 설명해줘 | Curate |
| 비슷한 작품 찾아줘 | Image RAG |
| 이거랑 저거 비교해줘 | Compare |

### 라우팅 방식

1. Rule-based  
2. Keyword Matching  
3. LLM Fallback  

---

# 🏗️ 프로젝트 구조

프로젝트는 다음과 같은 **Frontend / Backend / AI 구조**로 구성됩니다.
