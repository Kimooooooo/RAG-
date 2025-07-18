import streamlit as st
from main import load_vector_store,chain_prompt,classify_worldview
from langchain.memory import ConversationBufferMemory
import uuid
import pandas as pd
from langchain_core.runnables.history import RunnableWithMessageHistory

df = pd.read_csv('스팀게임파일.csv')

if 'session_id' not in st.session_state:
    st.session_state.session_id =str(uuid.uuid4()) #고유한 세션 식별자(UUIDv4)를 생성하여 저장
    # 3,5는 유저입력값이 잇을경우 / 1은 네트워크카드의 고유식별자를 유출당하면 그대로 해킹당함

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = load_vector_store() #벡터함수를 벡터DB사용 

if 'chain' not in st.session_state:
    st.session_state.chain,st.session_state.retriever = chain_prompt(st.session_state.vectorstore,session_id=st.session_state.session_id) #체인함수가져와서 chain 한 llm ,프롬포트 ,파서 기능을 사용함,retriever역시 따로 리터햇기때문에 지정해줘야함

if 'all_memory' not in st.session_state: #모든 대화기록 기억
    st.session_state.all_memory = {}
    

st.title('게임 추천을 위한 챗봇')

st.markdown("### 🎮 게임 추천 조건 선택")

# 장르
genre_option = st.multiselect(
    "장르 (최대 3개 선택 가능)",
    options=["액션", "RPG", "전략", "어드벤처","FPS","시뮬레이션"],
    max_selections=3
)

# 세계관
worldview_option = st.selectbox(
    "세계관 (선택 안 해도 됨)",
    options=["", "다크 판타지", "일반 판타지", "SF", "현대", "역사","중세"]
)

# 카테고리 (2D/3D)
category_option = st.selectbox(
    "카테고리 (선택 안 해도 됨)",
    options=["", "2D", "3D"]
)

candidate_games = df.copy()

if genre_option:
    genre_pattern = '|'.join(genre_option)
    candidate_games = candidate_games[candidate_games["genres"].str.contains(genre_pattern, na=False)]

if category_option:
    candidate_games = candidate_games[candidate_games["category"].str.contains(category_option, na=False)]

# 2. 세계관 선택 시 → 후보 게임 설명에 대해 AI 분류
if worldview_option:
    candidate_games["predicted_worldview"] = candidate_games["description_kr"].apply(classify_worldview)
    candidate_games = candidate_games[candidate_games["predicted_worldview"] == worldview_option]
query = st.text_input("🎤 어떤 게임을 찾고 계신가요?", placeholder="예: 엘든링 같은 게임 추천해줘!")

if query and not candidate_games.empty:
    # 5개 정도만 벡터 유사도 검색
    context = "\n".join(candidate_games["new_description"].fillna("").tolist()[:5])
    print(query)
    print(context)
    
    response = st.session_state.chain.invoke({
        "question": query,
        "context": context,
        
        
    },
    config = {'configurable' : {'session_id':st.session_state.session_id}})
    st.markdown("## 🎯 추천 결과")
    st.write(response)