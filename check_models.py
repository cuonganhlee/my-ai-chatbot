# check_models.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

def verify_api_key():
    """
    Hàm này sẽ tải API key, kết nối đến Google và liệt kê các model khả dụng.
    """
    print("--- Bắt đầu kiểm tra Google API Key ---")

    # 1. Tải các biến môi trường từ file .env
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    # 2. Kiểm tra xem API key có tồn tại trong file .env không
    if not api_key:
        print("\n[LỖI] Không tìm thấy biến GOOGLE_API_KEY trong file .env của bạn.")
        print("Vui lòng kiểm tra lại file .env và đảm bảo nó có dạng: GOOGLE_API_KEY=\"AIza...\"")
        return

    print("Đã tìm thấy API key trong file .env.")

    # 3. Cấu hình thư viện Google với API key
    try:
        genai.configure(api_key=api_key)
        print("Đã cấu hình thành công với Google Generative AI.")
    except Exception as e:
        print(f"\n[LỖI] Có lỗi xảy ra khi cấu hình với API key: {e}")
        return

    # 4. Gọi API để liệt kê các model
    try:
        print("\nĐang gọi API để lấy danh sách các model hỗ trợ 'generateContent'...")
        
        available_models = []
        for m in genai.list_models():
            # Kiểm tra xem model có hỗ trợ phương thức 'generateContent' không
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        if available_models:
            print("\n[THÀNH CÔNG] API Key hợp lệ! Các model bạn có thể sử dụng là:")
            for model_name in available_models:
                print(f"  - {model_name}")
            print("\nBạn có thể sử dụng 'gemini-pro' hoặc 'gemini-1.0-pro' trong code của mình.")
        else:
            print("\n[CẢNH BÁO] API Key có vẻ hợp lệ nhưng không tìm thấy model nào hỗ trợ 'generateContent'.")
            print("Điều này có thể do project trên Google Cloud của bạn chưa được cấp quyền. Hãy thử tạo một API key trong project mới.")

    except Exception as e:
        print(f"\n[LỖI] Đã xảy ra lỗi khi gọi API của Google. Điều này thường do API key không hợp lệ hoặc đã hết hạn.")
        print(f"Chi tiết lỗi: {e}")

if __name__ == "__main__":
    verify_api_key()