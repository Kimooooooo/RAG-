from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.long_context_reorder import LongContextReorder
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
import os
from example import few_shot_examples

set_llm_cache(SQLiteCache("chat_cache.db"))

api = os.getenv('OPENAI_API_KEY')

def load_vector_store():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',openai_api_key = api)
    vectorstore =Chroma(
        persist_directory='./games_db',
        collection_name='games',
        embedding_function=embeddings
    )
    return vectorstore
def chain_prompt(vectorstore):
    llm = ChatOpenAI(model = 'gpt-4.1')
    output_parser = StrOutputParser()

    tuple_fewshot_example = []
    for example in few_shot_examples:
        tuple_fewshot_example.append(('human',example['question']))
        tuple_fewshot_example.append(('ai',example['answer']))

    prompt = ChatPromptTemplate.from_messages([
        ('system','당신은 게임추천 전문가이며 다음 예시를 참조하여 뒤에는 게임의 장르,세계관,가격,설명 요약해서 같이 대답하고 게임의 스크린샷을 보여주세요'),
        *tuple_fewshot_example,
        MessagesPlaceholder(variable_name='history'),
        ('human',"질문:{question}\n\n참고 문서:\n{context}")
    ])
    if memory is None:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='history',
            input_key='question'
        )

    retriever = vectorstore.as_retriever(search_kwargs={'k':5})

    compressor = LLMChainExtractor.from_llm(llm)

    compressor_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        compressor = compressor
    )

    chain =ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        output_parser = output_parser,
        combine_docs_chain_kwargs ={
            'prompt':prompt
        }
    
    )
    
    return chain ,compressor_retriever

vectorstore = load_vector_store()
chain = chain_prompt(vectorstore)

# 사용자 질문을 준비
question = "저렴한 SF 게임 추천해줘"

# chain에 질문을 던짐
result = chain({"question": question})

# 답변 출력
print(result["answer"])
