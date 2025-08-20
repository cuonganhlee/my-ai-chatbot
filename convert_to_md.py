# convert_to_md.py (Phiên bản nâng cấp, sử dụng LibreOffice + Pandoc)
import os
import subprocess
import pypandoc
from tqdm import tqdm
import shutil

def find_soffice_path():
    """Tìm đường dẫn đến file thực thi soffice.exe của LibreOffice."""
    # Các đường dẫn phổ biến
    possible_paths = [
        "C:/Program Files/LibreOffice/program/soffice.exe",
        "C:/Program Files (x86)/LibreOffice/program/soffice.exe"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    # Nếu không tìm thấy, kiểm tra trong biến môi trường PATH
    soffice_in_path = shutil.which("soffice")
    if soffice_in_path:
        return soffice_in_path
        
    return None

def convert_docs_to_markdown(source_dir='data copy', delete_original=False):
    """
    Quét thư mục, chuyển đổi .doc -> .docx (dùng LibreOffice), 
    sau đó chuyển đổi .docx -> .md (dùng Pandoc).
    """
    print(f"--- Bắt đầu quá trình chuyển đổi trong thư mục '{source_dir}' ---")
    
    soffice_path = find_soffice_path()
    if not soffice_path:
        print("\n[LỖI] Không tìm thấy LibreOffice. Vui lòng cài đặt LibreOffice và đảm bảo nó nằm trong PATH hệ thống.")
        return

    # --- BƯỚC 1: Chuyển đổi .doc sang .docx ---
    print("\n--- Bước 1: Chuyển đổi .doc sang .docx bằng LibreOffice ---")
    doc_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.doc') and not file.startswith('~$'):
                doc_files.append(os.path.join(root, file))

    if doc_files:
        print(f"Tìm thấy {len(doc_files)} file .doc để chuyển đổi.")
        with tqdm(total=len(doc_files), desc="Chuyển đổi .doc", unit="file") as pbar:
            for file_path in doc_files:
                try:
                    # Lệnh để chạy LibreOffice ở chế độ headless (không giao diện)
                    # Nó sẽ tạo ra một file .docx trong cùng thư mục
                    subprocess.run(
                        [soffice_path, '--headless', '--convert-to', 'docx', '--outdir', os.path.dirname(file_path), file_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    if delete_original:
                        os.remove(file_path)
                except subprocess.CalledProcessError as e:
                    print(f"\nLỗi khi chuyển đổi file {os.path.basename(file_path)}: {e.stderr.decode('utf-8', errors='ignore')}")
                pbar.update(1)
    else:
        print("Không có file .doc nào cần chuyển đổi.")

    # --- BƯỚC 2: Chuyển đổi .docx sang .md ---
    print("\n--- Bước 2: Chuyển đổi .docx sang .md bằng Pandoc ---")
    docx_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.docx') and not file.startswith('~$'):
                docx_files.append(os.path.join(root, file))

    if not docx_files:
        print("Không tìm thấy file .docx nào để chuyển đổi.")
        return

    print(f"Tìm thấy {len(docx_files)} file .docx để chuyển đổi.")
    successful_conversions = 0
    failed_conversions = []

    with tqdm(total=len(docx_files), desc="Chuyển đổi .docx", unit="file") as pbar:
        for file_path in docx_files:
            try:
                output_path = os.path.splitext(file_path)[0] + '.md'
                pypandoc.convert_file(file_path, 'gfm', outputfile=output_path)
                successful_conversions += 1
                if delete_original:
                    os.remove(file_path)
            except Exception as e:
                failed_conversions.append((file_path, str(e)))
            pbar.update(1)

    # --- BÁO CÁO KẾT QUẢ ---
    print("\n--- BÁO CÁO CHUYỂN ĐỔI .docx -> .md ---")
    print(f"  - Chuyển đổi thành công: {successful_conversions} file")
    print(f"  - Chuyển đổi thất bại: {len(failed_conversions)} file")
    
    if failed_conversions:
        print("\n  Danh sách các file thất bại:")
        for path, error in failed_conversions:
            print(f"    - File: {path}")
            print(f"      Lỗi: {error[:100]}...")
            
    if delete_original and (doc_files or docx_files):
        print("\nĐã xóa các file .doc và .docx gốc đã được chuyển đổi thành công.")
        
    print("\n--- Quá trình hoàn tất ---")

if __name__ == "__main__":
    # Nhớ sao lưu thư mục 'data' trước khi chạy với delete_original=True
    convert_docs_to_markdown(source_dir='data copy', delete_original=True)