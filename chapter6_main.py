import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


CHAT_HISTORY_STORAGE = {}

os.environ['PINECONE_API_KEY'] = ""
os.environ['OPENAI_API_KEY'] = ""


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
async def root():
    return {'hello': 'world'}


@app.get('/chat')
async def chat(question: str, user_id: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    vectorstore = PineconeVectorStore(index_name="qna-rag", embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

    history_based_rag_qna_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, history_based_rag_qna_prompt_template)
    history_based_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = CHAT_HISTORY_STORAGE.get(user_id, [])
    ai_msg = history_based_rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])
    CHAT_HISTORY_STORAGE[user_id] = chat_history

    return ai_msg["answer"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)