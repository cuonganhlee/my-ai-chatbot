# debug_loader.py
from langchain_community.document_loaders import UnstructuredFileLoader

# CHỌN MỘT FILE BẠN NGHI NGỜ HOẶC MUỐN KIỂM TRA
file_path = "data/Cac SOP 2023/SOP Bo nhiem/SOP -QUY TRÌNH BỔ NHIỆM.doc" 

print(f"--- Đang kiểm tra file: {file_path} ---")

loader = UnstructuredFileLoader(
    file_path,
    autodetect_encoding=True,
    languages=["vie", "eng"]
)

try:
    document = loader.load()
    print("\n--- NỘI DUNG TRÍCH XUẤT THÀNH CÔNG ---")
    # In ra 1000 ký tự đầu tiên
    print(document[0].page_content[:1000]) 
    print("\n--- METADATA ---")
    print(document[0].metadata)
except Exception as e:
    print(f"\n--- CÓ LỖI KHI TẢI FILE ---")
    print(e)