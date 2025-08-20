# analyze_data.py
import os
from collections import defaultdict
import pandas as pd

def analyze_directory(path=r'D:\Program File\Documents\Lê Anh Cường\AI Chatbot\data'):
    """
    Quét thư mục dữ liệu, phân loại file theo định dạng và xác định các file "có vấn đề".
    """
    print(f"--- Bắt đầu phân tích thư mục '{path}' ---")
    
    file_stats = defaultdict(int)
    problem_files = {
        "temp_files": [],
        "unsupported_formats": [],
        "old_word_format": [],
        "scanned_pdfs": [] # Sẽ cần kiểm tra thủ công sau
    }
    
    supported_extensions = ['.docx', '.pdf', '.txt', '.md', '.html', '.pptx', '.xlsx']
    
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Bỏ qua các file ẩn hoặc file tạm
            if file.startswith('.') or file.startswith('~$'):
                problem_files["temp_files"].append(file_path)
                continue
            
            # Lấy phần mở rộng của file (ví dụ: .docx)
            ext = os.path.splitext(file)[1].lower()
            file_stats[ext] += 1
            
            # Phân loại các file cần chú ý
            if ext == '.doc':
                problem_files["old_word_format"].append(file_path)
            elif ext not in supported_extensions:
                problem_files["unsupported_formats"].append(file_path)

    print("\n--- THỐNG KÊ CÁC LOẠI FILE ---")
    # Sắp xếp để dễ nhìn
    sorted_stats = sorted(file_stats.items(), key=lambda item: item[1], reverse=True)
    stats_df = pd.DataFrame(sorted_stats, columns=['Định dạng', 'Số lượng'])
    print(stats_df.to_string(index=False))
    
    print("\n--- DANH SÁCH CÁC FILE CẦN XỬ LÝ ---")
    
    if problem_files["temp_files"]:
        print(f"\n[ƯU TIÊN 1 - CẦN XÓA] Tìm thấy {len(problem_files['temp_files'])} file tạm của Office:")
        for f in problem_files["temp_files"]:
            print(f"  - {f}")

    if problem_files["unsupported_formats"]:
        print(f"\n[ƯU TIÊN 2 - CẦN CHUYỂN ĐỔI/LOẠI BỎ] Tìm thấy {len(problem_files['unsupported_formats'])} file không được hỗ trợ:")
        for f in problem_files["unsupported_formats"]:
            print(f"  - {f}")

    if problem_files["old_word_format"]:
        print(f"\n[ƯU TIÊN 3 - NÊN CHUYỂN ĐỔI] Tìm thấy {len(problem_files['old_word_format'])} file Word định dạng cũ (.doc):")
        for f in problem_files["old_word_format"]:
            print(f"  - {f}")
            
    print("\n[LƯU Ý] Hãy kiểm tra thủ công các file PDF để xác định file nào là dạng ảnh (scanned).")
    print("\n--- Phân tích hoàn tất ---")

if __name__ == "__main__":
    # Cài đặt pandas nếu chưa có: pip install pandas
    analyze_directory()