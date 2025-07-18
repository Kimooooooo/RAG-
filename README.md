# 🎮 LangChain 기반 스팀 게임 추천 챗봇

이 프로젝트는 Steam 게임 데이터를 바탕으로 사용자의 자연어 쿼리(예: "엘든링 같은 게임 추천해줘!")에 대해 유사한 게임을 추천하는 AI 챗봇입니다. Streamlit 인터페이스를 제공하며, LangChain + OpenAI Embedding + Chroma 벡터 DB를 활용합니다.

---

## 📌 기능 요약

- ✅ 사용자 쿼리 벡터화 + 유사 게임 검색
- ✅ 장르/게임 타입/세계관 조건 필터링
- ✅ Markdown 기반 스크린샷 이미지 포함 출력
- ✅ 세계관 AI 분류 게임설명 을 통해 ai가 세계관을 추정하여 보여줌
- ✅ Streamlit UI 지원 (웹 기반 챗봇)
- ✅ 조건을 입력하면 조건을 최우선으로 유저의 인풋을 검색해줌

---

## 🧱 주요 구성 요소

### 1. 데이터 전처리
- 파일: `스팀게임파일_한글정제완료.csv`
- 주요 필드: `appid`, `name_kr`, `genres`, `category`, `description_kr`, `new_description`, `screenshots`

### 2. 벡터스토어 생성
- 사용 Embedding 모델: `text-embedding-3-large`
- 저장 위치: `./games_db`

### 3. LangChain 구성
- `ChatOpenAI` (`gpt-4.1`) 기반 LLM
- `ContextualCompressionRetriever` + `LLMChainExtractor`로 문서 압축 후 검색
- `RunnableWithMessageHistory`로 세션별 대화 기록 관리

### 4. UI 인터페이스 (Streamlit)
- 장르/세계관/타입 필터링 UI
- 챗봇 응답 출력
- 고려된 게임 리스트 함께 출력

---

## 🚀 실행 방법

1. **환경 준비**

```bash
git clone https://github.com/yourname/game-recommender-bot.git
cd game-recommender-bot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
