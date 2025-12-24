import os
import torch
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- LangChain Imports (Đã cập nhật chuẩn mới) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Cấu hình ---
class Settings:
    LLM_MODEL: str = "qwen2.5:7b"
    EMBEDDING_MODEL_ID: str = "bkai-foundation-models/vietnamese-bi-encoder"
    DATA_PATH: str = "./du_lieu_cong_ty.txt"
    DB_DIR: str = "./vector_db_bkai"
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

settings = Settings()

# Bộ nhớ chat (In-memory)
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class ChatRequest(BaseModel):
    query: str  
    session_id: str = "default_session"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Đang khởi động Server AI...")
    print(f"⚙️  Thiết bị: {settings.DEVICE.upper()}")

    # 1. Setup Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_ID,
        model_kwargs={'device': settings.DEVICE}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Setup Vector DB
    # Kiểm tra xem DB đã tồn tại và có dữ liệu không
    if os.path.exists(settings.DB_DIR) and os.listdir(settings.DB_DIR):
        print("📂 Đang tải Vector DB từ ổ cứng...")
        vector_db = Chroma(persist_directory=settings.DB_DIR, embedding_function=embedding_model)
    else:
        print("🔨 Đang tạo mới Vector DB...")
        if not os.path.exists(settings.DATA_PATH):
            # Tạo file mẫu nếu chưa có để tránh crash
            with open(settings.DATA_PATH, "w", encoding="utf-8") as f:
                f.write("Dữ liệu mẫu công ty VietCivil ID Solutions.")
            print(f"⚠️ Đã tạo file mẫu tại {settings.DATA_PATH}")
            
        loader = TextLoader(settings.DATA_PATH, encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100) # Tăng chunk size một chút
        chunks = text_splitter.split_documents(docs)
        vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=settings.DB_DIR)

    llm = OllamaLLM(model=settings.LLM_MODEL, temperature=0.1)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 4}) # Tăng k lên 4

    contextualize_q_system_prompt = (
        "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất của người dùng, "
        "hãy viết lại thành một câu hỏi độc lập có thể hiểu được mà không cần ngữ cảnh cũ. "
        "KHÔNG trả lời, chỉ viết lại câu hỏi hoặc giữ nguyên nếu đã rõ ràng."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
        Bạn là Trợ lý AI chuyên trách hỗ trợ người dùng sử dụng "Hệ thống Quản lý Dân cư". Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp trong phần ngữ cảnh (Context) dưới đây.

        Quy tắc trả lời:
        1. **Chỉ sử dụng thông tin trong Context:** Không được tự bịa ra thông tin, quy trình hoặc tính năng không có trong tài liệu được cung cấp. Nếu thông tin không có trong Context, hãy trả lời: "Xin lỗi, hiện tại tài liệu hướng dẫn chưa cập nhật thông tin về vấn đề này. Vui lòng liên hệ quản trị viên để được hỗ trợ."
        2. **Vai trò và Phân quyền:** Luôn lưu ý đến vai trò của người dùng (Tổ trưởng, Tổ phó, Cán bộ) nếu câu hỏi liên quan đến quyền hạn (ví dụ: xem báo cáo, thống kê).
        3. **Phong cách trình bày:**
        - Trả lời ngắn gọn, chuyên nghiệp, giọng văn hành chính nhưng thân thiện.
        - Nếu là quy trình các bước, hãy sử dụng gạch đầu dòng hoặc đánh số (1, 2, 3...) để dễ theo dõi.
        - Các tên nút bấm, tên menu, hoặc trạng thái (ví dụ: "Mới sinh", "Đã qua đời") nên được đặt trong dấu ngoặc kép hoặc in đậm để người dùng dễ nhận biết.
        4. **Xử lý tình huống:**
        - Nếu người dùng hỏi về "nhập liệu cho trẻ sơ sinh", hãy nhắc họ không cần điền nghề nghiệp/CCCD.
        - Nếu người dùng hỏi về "nhiều người cùng phản ánh", hãy hướng dẫn tính năng "Gộp kiến nghị".

        Ngữ cảnh (Context):
        {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Lưu chain vào app.state thay vì biến global
    app.state.final_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    print("✅ Server đã sẵn sàng phục vụ!")
    yield
    print("🛑 Server đang tắt...")

app = FastAPI(title="VietCivil ID Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.post("/chat_stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request):
    # Kiểm tra chain từ app.state
    if not hasattr(req.app.state, "final_chain"):
        raise HTTPException(status_code=503, detail="Hệ thống AI chưa sẵn sàng hoặc đang khởi động.")

    final_chain = req.app.state.final_chain

    async def generate_response() -> AsyncGenerator[str, None]:
        config = {"configurable": {"session_id": request.session_id}}
        
        try:
            async for chunk in final_chain.astream(
                {"input": request.query}, 
                config=config
            ):
                if "answer" in chunk:
                    # Trả về từng token/chunk
                    yield chunk["answer"]
        except Exception as e:
            yield f"\n[Lỗi hệ thống: {str(e)}]"
    
    return StreamingResponse(generate_response(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)