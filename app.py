import streamlit as st
from main import load_vector_store, chain_prompt, classify_worldview
import uuid
import pandas as pd
import re

# 장르 매핑 (한글 -> 영어)
GENRE_MAPPING = {
    "액션": "Action",
    "RPG": "RPG", 
    "전략": "Strategy",
    "어드벤처": "Adventure",
    "시뮬레이션": "Simulation",
    "스포츠": "Sports",
    "레이싱": "Racing",
    "퍼즐": "Puzzle",
    "인디": "Indie"
}

# CSV 파일 불러오기
@st.cache_data
def load_game_data():
    try:
        df = pd.read_csv('스팀게임파일_한글정제완료.csv')
        return df
    except:
        st.error("CSV 파일을 찾을 수 없습니다")
        st.stop()

df = load_game_data()

# 세션 상태 초기화
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = load_vector_store()

if 'chain' not in st.session_state:
    st.session_state.chain, st.session_state.retriever = chain_prompt(
        st.session_state.vectorstore,
        session_id=st.session_state.session_id
    )

# 🔧 스크린샷 및 설명 정리 함수
def extract_images_and_clean(text):
    if not isinstance(text, str):
        return ""
    
    # HTML 태그에서 이미지 추출
    image_urls = re.findall(r'<img[^>]+src="([^"]+)"', text)

    # 또는 스크린샷 URL 나열된 경우도 처리
    if not image_urls and 'http' in text:
        image_urls = re.findall(r'https?://[^\s|]+', text)

    markdown_images = [f"![게임 이미지]({url})" for url in image_urls]

    # HTML 제거
    clean_text = re.sub(r'<[^>]+>', '', text)

    return clean_text.strip() + "\n" + "\n".join(markdown_images)

# 🔧 설명 + 스크린샷 결합
def combine_description_and_screenshots(row):
    description = row.get("new_description", "")
    screenshots = row.get("screenshots", "")
    return f"{description}\n{screenshots}"

# UI
st.title('게임 추천 챗봇')

# 조건 선택
genre_option = st.multiselect(
    "장르 선택",
    options=["액션", "RPG", "전략", "어드벤처", "시뮬레이션", "스포츠", "레이싱", "퍼즐", "인디"],
    max_selections=3
)

worldview_option = st.selectbox(
    "세계관 선택",
    options=["", "다크 판타지", "일반 판타지", "SF", "현대", "역사", "중세"]
)

game_type_option = st.selectbox(
    "게임 타입 선택", 
    options=["", "멀티플레이어", "싱글플레이어", "PvP", "협동", "온라인"]
)

# 검색
with st.form(key="search_form"):
    query = st.text_input("게임 검색", placeholder="예: 엘든링 같은 게임 추천해줘!")
    submitted = st.form_submit_button("검색")

if submitted and query:
    candidate_games = df.copy()
    
    # 장르 필터링
    if genre_option:
        english_genres = [GENRE_MAPPING.get(genre, genre) for genre in genre_option]
        genre_pattern = '|'.join(english_genres)
        mask = candidate_games["genres"].str.contains(genre_pattern, na=False, case=False)
        candidate_games = candidate_games[mask]
    
    # 게임 타입 필터링
    if game_type_option:
        type_keywords = {
            "멀티플레이어": ["멀티", "Multiplayer"],
            "싱글플레이어": ["싱글", "Single-player"], 
            "PvP": ["PvP"],
            "협동": ["협동", "Co-op"],
            "온라인": ["온라인", "Online"]
        }
        keywords = type_keywords.get(game_type_option, [game_type_option])
        type_pattern = '|'.join(keywords)
        mask = candidate_games["category"].str.contains(type_pattern, na=False, case=False)
        candidate_games = candidate_games[mask]
    
    # 세계관 필터링 (AI 분석)
    if worldview_option:
        with st.spinner(f"'{worldview_option}' 세계관 게임을 AI가 분석하는 중... (시간이 걸릴 수 있습니다)"):
            games_to_analyze = candidate_games.head(100) if len(candidate_games) > 100 else candidate_games
            mask = games_to_analyze["description_kr"].notna()
            if mask.any():
                games_to_analyze.loc[mask, "predicted_worldview"] = games_to_analyze.loc[mask, "description_kr"].apply(classify_worldview)
                candidate_games = games_to_analyze[games_to_analyze["predicted_worldview"] == worldview_option]
    
    if candidate_games.empty:
        st.error("조건에 맞는 게임이 없습니다")
    else:
        # 🎯 스크린샷 포함 설명 정리 → context로 변환
        context_list = candidate_games.apply(combine_description_and_screenshots, axis=1).map(extract_images_and_clean).tolist()[:5]
        context = "\n".join([desc for desc in context_list if desc.strip()])
        
        if context.strip():
            with st.spinner("AI 추천 중..."):
                response = st.session_state.chain.invoke(
                    {"question": query, "context": context},
                    config={'configurable': {'session_id': st.session_state.session_id}}
                )
            
            # 결과 출력
            st.markdown("## 추천 결과")
            st.markdown(response, unsafe_allow_html=False)  # 이미지 포함해서 출력
            
            # 고려된 게임 리스트
            st.markdown("### 고려된 게임들")
            for _, row in candidate_games[['name_kr', 'genres']].head(5).iterrows():
                st.write(f"- {row['name_kr']} ({row['genres']})")
