import nest_asyncio
nest_asyncio.apply()
# app.py (Phiên bản chính xác cho LangChain v0.2.x trở lên)
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore # <<< THAY ĐỔI QUAN TRỌNG
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# --- PHẦN LOGIC (Cập nhật hoàn toàn theo cú pháp mới) ---
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Lấy tên index từ biến môi trường, nếu không có thì dùng tên mặc định
    index_name = os.getenv("PINECONE_INDEX_NAME", "my-chatbot-index-gemini")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, 
        embedding=embeddings
    )
    return vectorstore

def get_context_retriever_chain(vector_store):
    # Model này chỉ dùng để tạo câu hỏi tìm kiếm, có thể dùng model nhanh hơn
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0, convert_system_message_to_human=True)
    retriever = vector_store.as_retriever()
    
    # PROMPT ĐÚNG CHO VIỆC TẠO CÂU HỎI TÌM KIẾM
    # Nó chỉ nhận vào 'chat_history' và 'input', không có 'context'
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Dựa vào cuộc hội thoại trên, hãy tạo ra một câu hỏi tìm kiếm độc lập để có thể tìm thấy thông tin liên quan đến câu hỏi cuối cùng của người dùng.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    # Model này dùng để suy luận và trả lời, nên dùng model mạnh hơn
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.7, convert_system_message_to_human=True)
    
    # PROMPT ĐÚNG CHO VIỆC TẠO CÂU TRẢ LỜI CUỐI CÙNG
    # Prompt này mới là nơi cần biến 'context'
    system_prompt = (
        "Bạn là một trợ lý AI chuyên nghiệp, nhiệm vụ của bạn là trả lời các câu hỏi về tài liệu nội bộ. "
        "Hãy sử dụng các đoạn văn bản được cung cấp dưới đây để trả lời câu hỏi của người dùng. "
        "Trả lời một cách ngắn gọn, chính xác và đi thẳng vào vấn đề. "
        "Nếu các đoạn văn bản không chứa thông tin để trả lời câu hỏi, hãy nói rõ ràng rằng: "
        "'Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.' "
        "Tuyệt đối không được bịa ra thông tin không có trong văn bản. "
        "Hãy trả lời bằng tiếng Việt.\n\n"
        "Nội dung văn bản:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# --- PHẦN GIAO DIỆN STREAMLIT ---
def main():
    st.set_page_config(page_title="Chat với tài liệu của bạn", page_icon=":books:")
    st.header("Chat với tài liệu của bạn (dùng Gemini) :sparkles:")
    
    ADMIN_KEYWORD = "key_admin"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # --- TẠM THỜI BỎ CACHE VÀ THÊM LOG ĐỂ DEBUG ---
    st.write("Bắt đầu khởi tạo các thành phần...")

    try:
        vector_store = get_vectorstore()
        st.write("✅ Đã kết nối Vector Store thành công.")
        
        retriever_chain = get_context_retriever_chain(vector_store)
        st.write("✅ Đã tạo Retriever Chain thành công.")
        
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        st.write("✅ Đã tạo RAG Chain thành công.")
        
        admin_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
        st.write("✅ Đã khởi tạo Admin LLM thành công.")
        
        st.success("Tất cả các thành phần đã được khởi tạo thành công!")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình khởi tạo: {e}")
        # Dừng ứng dụng nếu có lỗi khởi tạo
        return

    # --- PHẦN CÒN LẠI CỦA HÀM MAIN GIỮ NGUYÊN ---
    
    # Hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    user_question = st.chat_input("Hãy hỏi tôi bất cứ điều gì về tài liệu của bạn:")

    if user_question:
        # ... (toàn bộ logic xử lý câu hỏi, chế độ admin, và hiển thị context giữ nguyên như cũ) ...
        # ... (copy phần đó từ phiên bản code trước của bạn) ...