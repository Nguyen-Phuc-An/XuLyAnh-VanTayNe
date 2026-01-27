"""
Chương trình chính - Hệ thống nhận dạng vân tay
"""

import sys
import os

# Set encoding cho Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

# Thêm đường dẫn để import các module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from giao_dien.giao_dien_chinh import tao_giao_dien


def main():
    """Hàm main"""
    print("=" * 60)
    print("HỆ THỐNG NHẬN DẠNG VÂN TAY")
    print("=" * 60)
    print("Bắt đầu khởi động giao diện...")
    
    try:
        tao_giao_dien()
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
