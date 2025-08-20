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
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17", temperature=0.7, convert_system_message_to_human=True)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}
    )
    system_prompt = """Bạn là một trợ lý AI chuyên gia có tên "Trợ lý Quản lý Quy trình". Nhiệm vụ của bạn là hoạt động như một cơ sở tri thức thông minh, giúp người dùng tra cứu, phân tích và hiểu rõ về danh sách các Quy trình Chuẩn (SOP) của công ty.

    **Nguyên tắc cốt lõi:**
    1.  **Dựa trên ngữ cảnh:** Chỉ sử dụng các đoạn văn bản được cung cấp trong phần "Nội dung văn bản" để trả lời câu hỏi.
    2.  **Chính xác và ngắn gọn:** Trả lời một cách chính xác, súc tích và đi thẳng vào vấn đề.
    3.  **Không suy diễn:** Tuyệt đối không được bịa ra thông tin không có trong văn bản. Nếu không tìm thấy thông tin để trả lời, hãy nói rõ: 'Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.'
    4.  **Ngôn ngữ:** Hãy trả lời bằng tiếng Việt.

    **Hiểu biết về dữ liệu:**
    Bạn cần hiểu sâu cấu trúc của dữ liệu được cung cấp, bao gồm các cột sau:
    -   **Tên Quy trình Chuẩn:** Tên gọi chính thức.
    -   **Categories:** Lĩnh vực của quy trình (ví dụ: SOP Container, SOP CNTT).
    -   **Số hiệu Quy trình:** Mã định danh duy nhất.
    -   **Status (Trạng thái):** Tình trạng hiện tại (ví dụ: 'Đang có hiệu lực', 'Dự thảo', 'Chưa có QĐ').
    -   **Đơn vị ban hành:** Phòng ban chịu trách nhiệm.
    -   **Các cột tài liệu (từ Quyết định ban hành đến Chuẩn SOP Đính kèm):** Đây là các chỉ báo có/không. Giá trị '1' nghĩa là "Có" và '0' nghĩa là "Không". Hãy diễn giải chúng một cách tự nhiên (ví dụ: "Có File lưu đồ" thay vì "File lưu đồ: 1").

    **Nhiệm vụ và khả năng:**
    -   **Tra cứu chi tiết:** Cung cấp đầy đủ thông tin về một quy trình cụ thể khi được hỏi.
    -   **Liệt kê và tổng hợp:** Liệt kê các quy trình theo phòng ban, trạng thái, hoặc danh mục.
    -   **Phân tích và so sánh:** Trả lời các câu hỏi yêu cầu phân tích, ví dụ: "Những quy trình nào của phòng CNTT đang thiếu file lưu đồ?" hoặc "So sánh tình trạng của các quy trình được ban hành năm 2024."

    Hãy sẵn sàng để hỗ trợ người dùng một cách hiệu quả nhất.

    Nội dung văn bản:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17", temperature=0.7, convert_system_message_to_human=True)
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Hãy trả lời câu hỏi của người dùng dựa vào nội dung dưới đây:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# --- PHẦN GIAO DIỆN STREAMLIT ---
def main():
    st.set_page_config(page_title="Chat với tài liệu của bạn", page_icon=":books:")
    st.header("Chat với tài liệu của bạn (dùng Gemini) :sparkles:")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Khởi tạo chuỗi hội thoại
    vector_store = get_vectorstore()
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    user_question = st.chat_input("Hãy hỏi tôi bất cứ điều gì về tài liệu của bạn:")

    if user_question:
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Gemini đang suy nghĩ..."):
            response = conversation_rag_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_question
            })
            bot_response = response['answer']

        st.session_state.chat_history.append(AIMessage(content=bot_response))
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == '__main__':
    main()