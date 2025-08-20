# ingest_data.py (Phiên bản cuối cùng, tự động phát hiện encoding và có thống kê)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader # <-- Thêm UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from tqdm import tqdm

load_dotenv()

# --- Phần khởi tạo giữ nguyên ---
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
INDEX_NAME = "my-chatbot-index-gemini"

def ingest_data():
    print("Bắt đầu quá trình nạp dữ liệu...")
    
    # --- BƯỚC THỐNG KÊ BAN ĐẦU ---
    print("1. Đang đếm tổng số file trong thư mục 'data'...")
    all_files = [os.path.join(root, file) for root, _, files in os.walk('data/') for file in files]
    # Lọc ra các file tạm của Office để không tính vào tổng số
    source_files = [f for f in all_files if not os.path.basename(f).startswith('~$')]
    total_files = len(source_files)
    print(f"   -> Tìm thấy tổng cộng {total_files} file nguồn.")

    # --- BƯỚC TẢI DỮ LIỆU CẢI TIẾN ---
    print("\n2. Đang quét và tải các tài liệu được hỗ trợ...")
    
    # THAY ĐỔI QUAN TRỌNG: Sử dụng UnstructuredFileLoader và bật autodetect_encoding
    loader = DirectoryLoader(
        'data/', 
        glob="**/*",
        show_progress=False, 
        # Chỉ định loader_cls và truyền tham số autodetect_encoding vào nó
        loader_cls=UnstructuredFileLoader,
        loader_kwargs={
            "autodetect_encoding": True, # <-- TỰ ĐỘNG PHÁT HIỆN ENCODING
            "languages": ["vie", "eng"]
        },
        silent_errors=True,
        use_multithreading=False 
    )
    
    documents = []
    with tqdm(total=total_files, desc="   -> Đang xử lý các file", unit="file") as pbar:
        for doc in loader.lazy_load():
            # Chỉ thêm vào danh sách nếu doc không rỗng
            # Bỏ [0] vì lazy_load trả về trực tiếp đối tượng Document
            if doc and doc.page_content.strip():
                documents.append(doc)
            pbar.update(1)

    # --- BƯỚC IN BÁO CÁO THỐNG KÊ ---
    successful_files = len(documents)
    failed_files = total_files - successful_files
    
    print("\n--- BÁO CÁO TẢI DỮ LIỆU ---")
    print(f"   - Tổng số file nguồn: {total_files}")
    print(f"   - Số file xử lý thành công: {successful_files}")
    print(f"   - Số file bị bỏ qua/lỗi: {failed_files}")
    print("--------------------------")

    if not documents:
        print("\n[KẾT THÚC] Không có tài liệu nào được xử lý thành công.")
        return

    print("\n3. Đang chia nhỏ tài liệu thành các chunks...") 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    print(f"   -> Đã chia thành {len(texts)} chunks văn bản.")

    # --- Phần tạo index và nạp dữ liệu giữ nguyên ---
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"\n4. Đang tạo index mới trên Pinecone: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("   -> Tạo index thành công.")
    else:
        print(f"\n4. Sử dụng index đã có: {INDEX_NAME}")

    print("\n5. Đang tạo embeddings và đẩy dữ liệu lên Pinecone...")
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="   -> Đang đẩy chunks", unit="batch"):
        i_end = min(i + batch_size, len(texts))
        batch = texts[i:i_end]
        PineconeVectorStore.from_documents(
            documents=batch, 
            embedding=embeddings,
            index_name=INDEX_NAME
        )
    
    print("\n[THÀNH CÔNG] Hoàn tất! Dữ liệu đã được nạp thành công lên Pinecone.")

if __name__ == "__main__":
    if INDEX_NAME in pc.list_indexes().names():
        print(f"Đang xóa index cũ: {INDEX_NAME}...")
        pc.delete_index(INDEX_NAME)
        print("Xóa thành công.")
    
    ingest_data()