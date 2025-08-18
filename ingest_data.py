# ingest_data.py (Phiên bản chính xác cho LangChain v0.2.x trở lên)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # <<< THAY ĐỔI QUAN TRỌNG 1
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore # <<< THAY ĐỔI QUAN TRỌNG 2
from pinecone import Pinecone as PineconeClient, ServerlessSpec

load_dotenv()

pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
INDEX_NAME = "my-chatbot-index-gemini"

def ingest_data():
    loader = DirectoryLoader('data/', glob="**/*", show_progress=True, use_multithreading=True, silent_errors=True)
    documents = loader.load()
    if not documents:
        print("Không tìm thấy tài liệu nào.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Đã chia thành {len(texts)} chunks văn bản.")

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Đang tạo index mới: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Tạo index thành công.")

    print("Đang tạo embeddings và thêm vào Pinecone...")
    # Cú pháp from_documents mới, sử dụng các tham số được đặt tên rõ ràng
    PineconeVectorStore.from_documents(
        documents=texts, 
        embedding=embeddings, # <<< THAY ĐỔI QUAN TRỌNG 3
        index_name=INDEX_NAME
    )
    print("Hoàn tất! Dữ liệu đã được nạp thành công lên Pinecone.")

if __name__ == "__main__":
    ingest_data()