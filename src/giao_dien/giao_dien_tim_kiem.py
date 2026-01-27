"""
Module giao diện tìm kiếm/nhận dạng người dùng
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tien_xu_ly.chuyen_xam import chuyen_nh_xam
from tien_xu_ly.chuan_hoa import chuan_hoa_anh
from tien_xu_ly.tang_cuong import ap_dung_gabor_filter
from phan_doan.nhi_phan_hoa import nhi_phan_hoa_otsu
from lam_manh.lam_manh_anh import lam_manh_scikit_image
from trich_dac_trung.trich_dac_trung_chi_tiet import trich_minutiae_chi_tiet


class GiaoDienTimKiem:
    """Giao diện tìm kiếm/nhận dạng người dùng"""
    
    def __init__(self, parent_frame, database_handler):
        """
        Khởi tạo giao diện tìm kiếm
        
        Args:
            parent_frame: Frame cha
            database_handler: Instance DatabaseEventHandler
        """
        self.parent = parent_frame
        self.db_handler = database_handler
        
        # Biến lưu trữ
        self.anh_duong_dan = None
        self.anh_goc = None
        self.anh_xam = None
        self.anh_xu_ly = None  # Ảnh tiền xử lý (chưa làm mảnh)
        self.anh_manh = None
        self.minutiae = None
        
        self._tao_giao_dien()
    
    def _tao_giao_dien(self):
        """Tạo giao diện tìm kiếm"""
        # Frame chính
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame chính sử dụng PanedWindow cho 3 cột
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ===== CỘT TRÁI: Tùy chọn tìm kiếm =====
        options_frame = ttk.Frame(paned_window)
        paned_window.add(options_frame, weight=1)
        
        # Label cho phần tùy chọn
        ttk.Label(options_frame, text="⚙️ TÙY CHỌN TÌM KIẾM", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Phương pháp so khớp
        ttk.Label(options_frame, text="Phương pháp:", font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=5)
        self.var_method = tk.StringVar(value="minutiae")
        combo_method = ttk.Combobox(options_frame, textvariable=self.var_method,
                                   values=["minutiae", "feature", "lbp", "ridge", "frequency"],
                                   state='readonly', width=30)
        combo_method.pack(pady=5, fill=tk.X, padx=5)
        
        # Separator
        ttk.Separator(options_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)
        
        # ===== Phần thông tin xử lý =====
        ttk.Label(options_frame, text="Trạng thái xử lý:", font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=5)
        
        self.text_status = tk.Text(options_frame, height=6, width=35, state=tk.DISABLED, wrap=tk.WORD)
        self.text_status.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== Buttons =====
        button_frame = ttk.Frame(options_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="🔍 Tìm Kiếm",
                  command=self._tim_kiem).pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
        
        ttk.Button(button_frame, text="🗑️ Xóa",
                  command=self._xoa_ket_qua).pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
        
        # ===== CỘT GIỮA: Ảnh xem trước =====
        image_frame = ttk.Frame(paned_window)
        paned_window.add(image_frame, weight=1)
        
        ttk.Label(image_frame, text="📸 HÌNH ẢNH TÌM KIẾM", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Canvas hiển thị ảnh
        self.canvas_image = tk.Canvas(image_frame, bg='#e0e0e0')
        self.canvas_image.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== CỘT PHẢI: Nút thao tác và kết quả =====
        action_frame = ttk.Frame(paned_window)
        paned_window.add(action_frame, weight=1)
        
        ttk.Label(action_frame, text="📁 QUẢN LÝ ẢNH & KẾT QUẢ", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Upload buttons
        button_top_frame = ttk.Frame(action_frame)
        button_top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_top_frame, text="📁 Chọn Ảnh", 
                  command=self._chon_anh).pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
        
        ttk.Button(button_top_frame, text="🔄 Xóa Ảnh", 
                  command=self._xoa_anh).pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
        
        # Separator
        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)
        
        # Kết quả header
        ttk.Label(action_frame, text="🎯 KẾT QUẢ", 
                 font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=5, pady=(10, 5))
        
        # Treeview kết quả
        columns = ("Họ Tên", "Username", "Điểm")
        self.tree_result = ttk.Treeview(action_frame, columns=columns, height=12)
        
        # Cấu hình cột
        self.tree_result.column('#0', width=0, stretch=tk.NO)
        self.tree_result.column("Họ Tên", anchor=tk.W, width=100)
        self.tree_result.column("Username", anchor=tk.W, width=80)
        self.tree_result.column("Điểm", anchor=tk.CENTER, width=60)
        
        # Header
        self.tree_result.heading('#0', text='', anchor=tk.W)
        self.tree_result.heading("Họ Tên", text="Họ Tên", anchor=tk.W)
        self.tree_result.heading("Username", text="Username", anchor=tk.W)
        self.tree_result.heading("Điểm", text="Điểm (%)", anchor=tk.CENTER)
        
        self.tree_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _chon_anh(self):
        """Chọn ảnh vân tay"""
        duong_dan = filedialog.askopenfilename(
            title="Chọn ảnh vân tay để tìm kiếm",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp"), ("Tất cả", "*.*")]
        )
        
        if not duong_dan:
            return
        
        try:
            self.anh_duong_dan = duong_dan
            
            # Đọc ảnh
            self.anh_goc, self.anh_xam = chuyen_nh_xam(duong_dan)
            
            # Hiển thị ảnh toàn màn hình
            # Lấy kích thước canvas
            canvas_width = self.canvas_image.winfo_width()
            canvas_height = self.canvas_image.winfo_height()
            
            # Nếu canvas chưa có kích thước, dùng mặc định
            if canvas_width <= 1:
                canvas_width = 400
            if canvas_height <= 1:
                canvas_height = 400
            
            # Resize ảnh để vừa canvas
            aspect_ratio = self.anh_xam.shape[1] / self.anh_xam.shape[0]
            if (canvas_width / canvas_height) > aspect_ratio:
                # Canvas rộng hơn
                display_height = canvas_height
                display_width = int(display_height * aspect_ratio)
            else:
                # Canvas cao hơn
                display_width = canvas_width
                display_height = int(display_width / aspect_ratio)
            
            anh_display = cv2.resize(self.anh_xam, (display_width, display_height))
            anh_pil = Image.fromarray(anh_display)
            anh_tk = ImageTk.PhotoImage(anh_pil)
            
            # Xóa ảnh cũ
            self.canvas_image.delete("all")
            
            # Vẽ ảnh ở giữa canvas
            self.canvas_image.create_image(canvas_width // 2, canvas_height // 2, 
                                          image=anh_tk)
            self.canvas_image.image = anh_tk
            
            self._cap_nhat_trang_thai(f"✓ Đã chọn ảnh: {os.path.basename(duong_dan)}")
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi chọn ảnh: {str(e)}")
            self._cap_nhat_trang_thai(f"✗ Lỗi: {str(e)}")
    
    def _xoa_anh(self):
        """Xóa ảnh đã chọn"""
        self.anh_duong_dan = None
        self.anh_goc = None
        self.anh_xam = None
        self.anh_xu_ly = None
        self.anh_manh = None
        self.minutiae = None
        
        self.canvas_image.delete("all")
        self._cap_nhat_trang_thai("✓ Ảnh đã được xóa")
    
    def _xu_ly_anh(self):
        """Xử lý ảnh tự động"""
        if self.anh_xam is None:
            return False
        
        try:
            self._cap_nhat_trang_thai("⏳ Đang xử lý ảnh...")
            
            method = self.var_method.get()
            
            # Chuẩn hóa
            anh_chuan_hoa = chuan_hoa_anh(self.anh_xam)
            
            # Tăng cường
            anh_tang_cuong = ap_dung_gabor_filter(anh_chuan_hoa)
            
            # Nhị phân hóa
            anh_nhi_phan, _ = nhi_phan_hoa_otsu(anh_tang_cuong)
            
            # Lưu ảnh tiền xử lý (trước khi làm mảnh)
            self.anh_xu_ly = anh_nhi_phan.copy()
            
            # Làm mảnh
            self.anh_manh = lam_manh_scikit_image(anh_nhi_phan)
            
            # Trích minutiae (chỉ cần cho minutiae/comprehensive methods)
            if method in ['minutiae', 'comprehensive', 'feature']:
                self.minutiae = trich_minutiae_chi_tiet(self.anh_manh)
                minutiae_count = len(self.minutiae.get('endings', [])) + len(self.minutiae.get('bifurcations', []))
                status_msg = f"✓ Xử lý hoàn tất!\n  - Minutiae tìm thấy: {minutiae_count}"
            else:
                self.minutiae = None
                status_msg = "✓ Xử lý hoàn tát!"
            
            self._cap_nhat_trang_thai(status_msg)
            return True
        
        except Exception as e:
            self._cap_nhat_trang_thai(f"✗ Lỗi xử lý: {str(e)}")
            messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh: {str(e)}")
            return False
    
    def _tim_kiem(self):
        """Tìm kiếm người dùng"""
        if self.anh_xam is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh vân tay!")
            return
        
        # Kiểm tra kết nối database
        if not self.db_handler.kiểm_tra_kết_nối():
            return
        
        try:
            # Xử lý ảnh
            if not self._xu_ly_anh():
                return
            
            # Nhận dạng
            method = self.var_method.get()
            method_display = {
                'minutiae': 'Minutiae Matching',
                'feature': 'Feature Matching',
                'harris': 'Harris Corners',
                'orb': 'ORB Features',
                'lbp': 'LBP Texture',
                'ridge': 'Ridge Orientation',
                'frequency': 'Frequency Domain'
            }.get(method, method)
            
            self._cap_nhat_trang_thai(f"⏳ Đang tìm kiếm trong database ({method_display})...")
            
            results = self.db_handler.nhan_dang_van_tay(
                self.anh_manh, self.minutiae, method, anh_xu_ly=self.anh_xu_ly
            )
            
            # Xóa kết quả cũ
            self._xoa_ket_qua()
            
            # Hiển thị kết quả
            if results:
                # Tìm kết quả có điểm cao nhất
                best_score = max([r.get('similarity_score', 0) for r in results]) if results else 0
                
                for i, result in enumerate(results):
                    score = result.get('similarity_score', 0)
                    # Tô màu xanh cho kết quả có điểm cao nhất, nếu điểm > 70 cũng tô
                    is_best = (score == best_score)
                    is_match = (score > 70)
                    tag = ()
                    if is_best:
                        tag = ('best_match',)
                    elif is_match:
                        tag = ('match',)
                    
                    self.tree_result.insert(
                        "",
                        tk.END,
                        values=(
                            result.get('full_name', 'N/A'),
                            result.get('username', 'N/A'),
                            f"{score:.2f}"
                        ),
                        tags=tag if tag else ()
                    )
                
                # Cấu hình màu cho match
                self.tree_result.tag_configure('match', background='lightgreen')
                self.tree_result.tag_configure('best_match', background='lightgreen', foreground='darkgreen')
                
                self._cap_nhat_trang_thai(
                    f"✓ Tìm kiếm hoàn tất!\n"
                    f"  - Kết quả cao nhất: {best_score:.2f}%"
                )
            else:
                self._cap_nhat_trang_thai("✗ Không tìm thấy kết quả")
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi tìm kiếm: {str(e)}")
            self._cap_nhat_trang_thai(f"✗ Lỗi: {str(e)}")
    
    def _xoa_ket_qua(self):
        """Xóa kết quả tìm kiếm"""
        for item in self.tree_result.get_children():
            self.tree_result.delete(item)
    
    def _cap_nhat_trang_thai(self, message):
        """Cập nhật trạng thái"""
        self.text_status.config(state=tk.NORMAL)
        self.text_status.insert(tk.END, message + "\n")
        self.text_status.see(tk.END)
        self.text_status.config(state=tk.DISABLED)
