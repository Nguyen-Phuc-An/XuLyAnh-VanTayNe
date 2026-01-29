"""
Module hiển thị kết quả giao diện
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class HienThiKetQua:
    """Lớp hiển thị kết quả xử lý"""
    
    def __init__(self, root):
        self.root = root
        
        # Tạo style cho giao diện
        self._setup_style()
        
        # Tạo PanedWindow để chia thành 2 phần (ảnh và thông tin)
        paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === PHẦN 1: ẢNH (Bên trái) ===
        frame_anh_container = ttk.Frame(paned_window)
        paned_window.add(frame_anh_container, weight=10)
        
        # Notebook cho các ảnh (tabs)
        self.notebook = ttk.Notebook(frame_anh_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Ảnh gốc (Hiển thị cả 2 ảnh)
        tab_anh_goc = ttk.Frame(self.notebook)
        self.notebook.add(tab_anh_goc, text="Ảnh gốc")
        
        frame_2_anh_goc = ttk.Frame(tab_anh_goc)
        frame_2_anh_goc.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        frame_anh_goc_1 = ttk.LabelFrame(frame_2_anh_goc, text="Ảnh 1", padding=5)
        frame_anh_goc_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        self.canvas_anh_goc = tk.Canvas(frame_anh_goc_1, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_goc.pack(fill=tk.BOTH, expand=True)
        self.image_anh_goc = None
        
        frame_anh_goc_2 = ttk.LabelFrame(frame_2_anh_goc, text="Ảnh 2", padding=5)
        frame_anh_goc_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        self.canvas_anh_goc_2 = tk.Canvas(frame_anh_goc_2, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_goc_2.pack(fill=tk.BOTH, expand=True)
        self.image_anh_goc_2 = None
        
        # Tab 2: Ảnh sau xử lý (Hiển thị cả 2 ảnh)
        tab_anh_sau = ttk.Frame(self.notebook)
        self.notebook.add(tab_anh_sau, text="Sau xử lý")
        
        frame_2_anh_sau = ttk.Frame(tab_anh_sau)
        frame_2_anh_sau.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        frame_anh_sau_1 = ttk.LabelFrame(frame_2_anh_sau, text="Ảnh 1", padding=5)
        frame_anh_sau_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        self.canvas_anh_sau = tk.Canvas(frame_anh_sau_1, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_sau.pack(fill=tk.BOTH, expand=True)
        self.image_anh_sau = None
        
        frame_anh_sau_2 = ttk.LabelFrame(frame_2_anh_sau, text="Ảnh 2", padding=5)
        frame_anh_sau_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        self.canvas_anh_sau_2 = tk.Canvas(frame_anh_sau_2, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_sau_2.pack(fill=tk.BOTH, expand=True)
        self.image_anh_sau_2 = None
        
        # Tab 3: Minutiae (Hiển thị cả 2 ảnh)
        tab_anh_minutiae = ttk.Frame(self.notebook)
        self.tab_minutiae_index = self.notebook.add(tab_anh_minutiae, text="Chi tiết Minutiae")
        
        frame_2_anh_minutiae = ttk.Frame(tab_anh_minutiae)
        frame_2_anh_minutiae.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        frame_anh_minutiae_1 = ttk.LabelFrame(frame_2_anh_minutiae, text="Ảnh 1", padding=5)
        frame_anh_minutiae_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        self.canvas_anh_minutiae = tk.Canvas(frame_anh_minutiae_1, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_minutiae.pack(fill=tk.BOTH, expand=True)
        self.image_anh_minutiae = None
        
        frame_anh_minutiae_2 = ttk.LabelFrame(frame_2_anh_minutiae, text="Ảnh 2", padding=5)
        frame_anh_minutiae_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        self.canvas_anh_minutiae_2 = tk.Canvas(frame_anh_minutiae_2, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_minutiae_2.pack(fill=tk.BOTH, expand=True)
        self.image_anh_minutiae_2 = None
        
        # === PHẦN 2: THÔNG TIN (Bên phải) ===
        frame_info_container = ttk.Frame(paned_window)
        paned_window.add(frame_info_container, weight=1)
        
        # Nút tải ảnh ở trên cùng
        button_frame = ttk.Frame(frame_info_container)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(button_frame, text="Tải ảnh:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        # Note: Các button sẽ được set command từ xu_ly_su_kien sau
        self.btn_anh_1 = ttk.Button(button_frame, text="Ảnh 1", width=12)
        self.btn_anh_1.pack(side=tk.LEFT, padx=3)
        
        self.btn_anh_2 = ttk.Button(button_frame, text="Ảnh 2", width=12)
        self.btn_anh_2.pack(side=tk.LEFT, padx=3)
        
        # Tạo canvas scrollable
        canvas_scroll = tk.Canvas(frame_info_container, bg="#ecf0f1", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame_info_container, orient=tk.VERTICAL, command=canvas_scroll.yview)
        self.frame_info = ttk.Frame(canvas_scroll, style='Info.TFrame')
        
        self.frame_info.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        
        canvas_scroll.create_window((0, 0), window=self.frame_info, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Thông tin ảnh - Card style
        frame_anh = ttk.Frame(self.frame_info, style='Card.TFrame')
        frame_anh.pack(fill=tk.X, padx=8, pady=8)
        
        lbl_anh = ttk.Label(frame_anh, text="THÔNG TIN ẢNH", style='Title.TLabel')
        lbl_anh.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        frame_anh_content = ttk.Frame(frame_anh, style='Card.TFrame')
        frame_anh_content.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        self.label_kich_thuoc = ttk.Label(frame_anh_content, text="Kích thước: N/A", foreground='#27ae60')
        self.label_kich_thuoc.pack(anchor=tk.W, pady=3)
        
        # Thông tin chi tiết - Card style (nội dung sẽ thay đổi theo phương pháp)
        self.frame_details = ttk.Frame(self.frame_info, style='Card.TFrame')
        self.frame_details.pack(fill=tk.X, padx=8, pady=8)
        
        self.lbl_details = ttk.Label(self.frame_details, text="MINUTIAE", style='Title.TLabel')
        self.lbl_details.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        frame_details_content = ttk.Frame(self.frame_details, style='Card.TFrame')
        frame_details_content.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        # Minutiae labels - Image 1
        self.label_minutiae_img1_title = ttk.Label(frame_details_content, text="Ảnh 1:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_minutiae_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_minutiae_img1_title.grid_remove()
        
        self.label_ending = ttk.Label(frame_details_content, text="  Kết thúc: 0", foreground='#3498db')
        self.label_ending.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_ending.grid_remove()
        
        self.label_bifurcation = ttk.Label(frame_details_content, text="  Phân nhánh: 0", foreground='#3498db')
        self.label_bifurcation.grid(row=2, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_bifurcation.grid_remove()
        
        self.label_total = ttk.Label(frame_details_content, text="  Tổng: 0", foreground='#3498db')
        self.label_total.grid(row=3, column=0, sticky=tk.W, pady=3, padx=0)
        self.label_total.grid_remove()
        
        # Minutiae labels - Image 2
        self.label_minutiae_img2_title = ttk.Label(frame_details_content, text="Ảnh 2:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_minutiae_img2_title.grid(row=4, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_minutiae_img2_title.grid_remove()
        
        self.label_ending2 = ttk.Label(frame_details_content, text="  Kết thúc: 0", foreground='#e74c3c')
        self.label_ending2.grid(row=5, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_ending2.grid_remove()
        
        self.label_bifurcation2 = ttk.Label(frame_details_content, text="  Phân nhánh: 0", foreground='#e74c3c')
        self.label_bifurcation2.grid(row=6, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_bifurcation2.grid_remove()
        
        self.label_total2 = ttk.Label(frame_details_content, text="  Tổng: 0", foreground='#e74c3c')
        self.label_total2.grid(row=7, column=0, sticky=tk.W, pady=3, padx=0)
        self.label_total2.grid_remove()
        
        # Feature Matching labels
        self.label_feature_img1_title = ttk.Label(frame_details_content, text="Ảnh 1:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_feature_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_feature_img1_title.grid_remove()
        
        self.label_feature_count1 = ttk.Label(frame_details_content, text="  Đặc trưng: 0", foreground='#3498db')
        self.label_feature_count1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_feature_count1.grid_remove()
        
        self.label_feature_img2_title = ttk.Label(frame_details_content, text="Ảnh 2:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_feature_img2_title.grid(row=2, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_feature_img2_title.grid_remove()
        
        self.label_feature_count2 = ttk.Label(frame_details_content, text="  Đặc trưng: 0", foreground='#e74c3c')
        self.label_feature_count2.grid(row=3, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_feature_count2.grid_remove()
        
        self.label_good_matches = ttk.Label(frame_details_content, text="Khớp tốt: 0", foreground='#f39c12')
        self.label_good_matches.grid(row=4, column=0, sticky=tk.W, pady=3, padx=0)
        self.label_good_matches.grid_remove()
        
        # LBP labels
        self.label_lbp_img1_title = ttk.Label(frame_details_content, text="Ảnh 1:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_lbp_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_lbp_img1_title.grid_remove()
        
        self.label_lbp_histogram1 = ttk.Label(frame_details_content, text="  Histogram: -", foreground='#3498db')
        self.label_lbp_histogram1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_lbp_histogram1.grid_remove()
        
        self.label_lbp_img2_title = ttk.Label(frame_details_content, text="Ảnh 2:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_lbp_img2_title.grid(row=2, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_lbp_img2_title.grid_remove()
        
        self.label_lbp_histogram2 = ttk.Label(frame_details_content, text="  Histogram: -", foreground='#e74c3c')
        self.label_lbp_histogram2.grid(row=3, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_lbp_histogram2.grid_remove()
        
        self.label_lbp_distance = ttk.Label(frame_details_content, text="Khoảng cách Chi-square: 0.0000", foreground='#f39c12')
        self.label_lbp_distance.grid(row=4, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_lbp_distance.grid_remove()
        
        self.label_lbp_similarity = ttk.Label(frame_details_content, text="Tương đồng: 0.00%", foreground='#f39c12')
        self.label_lbp_similarity.grid(row=5, column=0, sticky=tk.W, pady=3, padx=0)
        self.label_lbp_similarity.grid_remove()
        
        # Ridge Orientation labels
        self.label_ridge_img1_title = ttk.Label(frame_details_content, text="Ảnh 1:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_ridge_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_ridge_img1_title.grid_remove()
        
        self.label_ridge_orientation1 = ttk.Label(frame_details_content, text="  Góc trung bình: -", foreground='#3498db')
        self.label_ridge_orientation1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_ridge_orientation1.grid_remove()
        
        self.label_ridge_img2_title = ttk.Label(frame_details_content, text="Ảnh 2:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_ridge_img2_title.grid(row=2, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_ridge_img2_title.grid_remove()
        
        self.label_ridge_orientation2 = ttk.Label(frame_details_content, text="  Góc trung bình: -", foreground='#e74c3c')
        self.label_ridge_orientation2.grid(row=3, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_ridge_orientation2.grid_remove()
        
        self.label_ridge_diff = ttk.Label(frame_details_content, text="Chênh lệch góc: 0.00°", foreground='#f39c12')
        self.label_ridge_diff.grid(row=4, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_ridge_diff.grid_remove()
        
        self.label_ridge_consistency = ttk.Label(frame_details_content, text="Độ nhất quán: 0.0000", foreground='#f39c12')
        self.label_ridge_consistency.grid(row=5, column=0, sticky=tk.W, pady=3, padx=0)
        self.label_ridge_consistency.grid_remove()
        
        # Frequency Domain labels
        self.label_freq_img1_title = ttk.Label(frame_details_content, text="Ảnh 1:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_freq_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_freq_img1_title.grid_remove()
        
        self.label_freq_fft1 = ttk.Label(frame_details_content, text="  FFT: -", foreground='#3498db')
        self.label_freq_fft1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_freq_fft1.grid_remove()
        
        self.label_freq_img2_title = ttk.Label(frame_details_content, text="Ảnh 2:", foreground='#2c3e50', font=('Arial', 9, 'bold'))
        self.label_freq_img2_title.grid(row=2, column=0, sticky=tk.W, pady=(5, 2), padx=0)
        self.label_freq_img2_title.grid_remove()
        
        self.label_freq_fft2 = ttk.Label(frame_details_content, text="  FFT: -", foreground='#e74c3c')
        self.label_freq_fft2.grid(row=3, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_freq_fft2.grid_remove()
        
        self.label_freq_spectrum = ttk.Label(frame_details_content, text="Phổ tần: 0.00%", foreground='#f39c12')
        self.label_freq_spectrum.grid(row=4, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_freq_spectrum.grid_remove()
        
        self.label_freq_energy = ttk.Label(frame_details_content, text="Năng lượng: 0.00%", foreground='#f39c12')
        self.label_freq_energy.grid(row=5, column=0, sticky=tk.W, pady=2, padx=0)
        self.label_freq_energy.grid_remove()
        
        self.label_freq_similarity = ttk.Label(frame_details_content, text="Tương đồng: 0.00%", foreground='#f39c12')
        self.label_freq_similarity.grid(row=6, column=0, sticky=tk.W, pady=3, padx=0)
        self.label_freq_similarity.grid_remove()
        
        # Thông tin so khớp - Card style
        frame_match = ttk.Frame(self.frame_info, style='Card.TFrame')
        frame_match.pack(fill=tk.X, padx=8, pady=8)
        
        lbl_match = ttk.Label(frame_match, text="KẾT QUẢ SO KHỚP", style='Title.TLabel')
        lbl_match.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        frame_match_content = ttk.Frame(frame_match, style='Card.TFrame')
        frame_match_content.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        self.label_match = ttk.Label(frame_match_content, text="Khớp: N/A", foreground='#9b59b6')
        self.label_match.pack(anchor=tk.W, pady=2)
        
        self.label_similarity = ttk.Label(frame_match_content, text="Tương đồng: N/A", foreground='#1abc9c')
        self.label_similarity.pack(anchor=tk.W, pady=3)
        
        # Label cảnh báo nếu Khớp thấp nhưng Tương đồng cao
        self.label_warning = ttk.Label(frame_match_content, text="", foreground='#ff6600')
        self.label_warning.pack(anchor=tk.W, pady=2)
        
        # Thông báo thành công - Card style
        frame_notification = ttk.Frame(self.frame_info, style='Card.TFrame')
        frame_notification.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        lbl_thong_bao = ttk.Label(frame_notification, text="THÔNG BÁO", style='Title.TLabel')
        lbl_thong_bao.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        # Text widget để hiển thị thông báo
        frame_thong_bao = ttk.Frame(frame_notification, style='Card.TFrame')
        frame_thong_bao.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        
        self.text_thong_bao = tk.Text(frame_thong_bao, height=10, width=30, font=("Consolas", 9), 
                                      bg="#ecf0f1", fg="#27ae60", wrap=tk.WORD, state=tk.DISABLED,
                                      relief=tk.SOLID, borderwidth=1, insertbackground='#27ae60')
        scrollbar_thong_bao = ttk.Scrollbar(frame_thong_bao, orient=tk.VERTICAL, command=self.text_thong_bao.yview)
        self.text_thong_bao.configure(yscrollcommand=scrollbar_thong_bao.set)
        
        self.text_thong_bao.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_thong_bao.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_style(self):
        """Cấu hình style cho giao diện"""
        style = ttk.Style()
        
        # Cấu hình style Info.TFrame với màu nền gradient
        style.configure('Info.TFrame', background='#ecf0f1')
        
        # Cấu hình style Card.TFrame với màu đẹp
        style.configure('Card.TFrame', background='#ffffff', relief='solid', borderwidth=1)
        
        # Cấu hình style Title.TLabel
        style.configure('Title.TLabel', background='#ffffff', foreground='#2c3e50', font=('Arial', 10, 'bold'))
        
        # Cấu hình các tab - giữ kích thước cố định
        style.configure('TNotebook', background='#ecf0f1')
        style.configure('TNotebook.Tab', padding=[30, 12], font=('Arial', 10), width=20)
        style.map('TNotebook.Tab',
                  background=[('selected', '#ffffff'), ('!selected', '#e8e8e8')],
                  foreground=[('selected', '#2c3e50'), ('!selected', '#7f8c8d')],
                  padding=[('selected', [30, 12]), ('!selected', [30, 12])])
        
        # Cấu hình các frame
        style.configure('TFrame', background='#ecf0f1')
        style.configure('TLabel', background='#ecf0f1', foreground='#2c3e50')
    
    def hien_thi_anh_goc(self, anh, anh_2=None):
        """Hiển thị ảnh gốc (hỗ trợ cả 2 ảnh)"""
        self._hien_thi_anh_len_canvas(anh, self.canvas_anh_goc)
        if anh_2 is not None:
            self._hien_thi_anh_len_canvas(anh_2, self.canvas_anh_goc_2)
    
    def hien_thi_anh_sau_xu_ly(self, anh, anh_2=None):
        """Hiển thị ảnh sau xử lý (hỗ trợ cả 2 ảnh)"""
        self._hien_thi_anh_len_canvas(anh, self.canvas_anh_sau)
        if anh_2 is not None:
            self._hien_thi_anh_len_canvas(anh_2, self.canvas_anh_sau_2)
    
    def hien_thi_anh_after_xu_ly(self, anh, anh_2=None):
        """Hiển thị ảnh minutiae (hỗ trợ cả 2 ảnh)"""
        self._hien_thi_anh_len_canvas(anh, self.canvas_anh_minutiae)
        if anh_2 is not None:
            self._hien_thi_anh_len_canvas(anh_2, self.canvas_anh_minutiae_2)
    
    def _hien_thi_anh_len_canvas(self, anh, canvas):
        """Hỗ trợ hiển thị ảnh lên canvas"""
        if anh is None:
            return
        
        # Lấy kích thước canvas
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Nếu canvas chưa render, dùng kích thước mặc định
            canvas_width = 600
            canvas_height = 500
        
        # Resize ảnh để vừa với canvas
        h, w = anh.shape[:2]
        ratio = min((canvas_width - 10) / w, (canvas_height - 10) / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        anh_resize = cv2.resize(anh, (new_w, new_h))
        
        # Chuyển đổi định dạng
        if len(anh_resize.shape) == 2:
            anh_rgb = cv2.cvtColor(anh_resize, cv2.COLOR_GRAY2RGB)
        else:
            anh_rgb = cv2.cvtColor(anh_resize, cv2.COLOR_BGR2RGB)
        
        # Chuyển thành PIL Image
        image_pil = Image.fromarray(anh_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Hiển thị lên canvas (giữa canvas)
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        canvas.create_image(x_center, y_center, image=image_tk)
        canvas.image = image_tk
    
    def cap_nhat_chi_tiet_minutiae(self, ending_count1, bifurcation_count1, total_count1, ending_count2=0, bifurcation_count2=0, total_count2=0):
        """Cập nhật thông tin Minutiae cho cả 2 ảnh"""
        # Image 1
        self.label_ending.config(text=f"  Kết thúc: {ending_count1}")
        self.label_bifurcation.config(text=f"  Phân nhánh: {bifurcation_count1}")
        self.label_total.config(text=f"  Tổng: {total_count1}")
        
        # Image 2
        if ending_count2 > 0 or bifurcation_count2 > 0 or total_count2 > 0:
            self.label_ending2.config(text=f"  Kết thúc: {ending_count2}")
            self.label_bifurcation2.config(text=f"  Phân nhánh: {bifurcation_count2}")
            self.label_total2.config(text=f"  Tổng: {total_count2}")
    
    def cap_nhat_chi_tiet_feature(self, feature_count1, feature_count2, good_matches):
        """Cập nhật thông tin Feature Matching cho cả 2 ảnh"""
        self.label_feature_count1.config(text=f"  Đặc trưng: {feature_count1}")
        self.label_feature_count2.config(text=f"  Đặc trưng: {feature_count2}")
        self.label_good_matches.config(text=f"Khớp tốt: {good_matches}")
    
    def cap_nhat_chi_tiet_lbp(self, chi_square_distance, histogram1_size=None, histogram2_size=None):
        """Cập nhật thông tin LBP Texture"""
        if histogram1_size and histogram1_size > 0:
            self.label_lbp_histogram1.config(text=f"  Histogram: {histogram1_size} bins")
        if histogram2_size and histogram2_size > 0:
            self.label_lbp_histogram2.config(text=f"  Histogram: {histogram2_size} bins")
        self.label_lbp_distance.config(text=f"Khoảng cách Chi-square: {chi_square_distance:.4f}")
        # Chi-square thấp = tương đồng cao
        similarity = max(0, 100 - (chi_square_distance * 10))
        self.label_lbp_similarity.config(text=f"Tương đồng: {similarity:.2f}%")
    
    def cap_nhat_chi_tiet_ridge(self, mean_angle_diff, avg_angle1=None, avg_angle2=None):
        """Cập nhật thông tin Ridge Orientation"""
        if avg_angle1 is not None:
            self.label_ridge_orientation1.config(text=f"  Góc trung bình: {avg_angle1:.2f}°")
        if avg_angle2 is not None:
            self.label_ridge_orientation2.config(text=f"  Góc trung bình: {avg_angle2:.2f}°")
        self.label_ridge_diff.config(text=f"Chênh lệch góc: {mean_angle_diff:.2f}°")
        # Tính độ nhất quán từ angle diff
        consistency = max(0, 100 - (mean_angle_diff * 2))
        self.label_ridge_consistency.config(text=f"Độ nhất quán: {consistency:.2f}%")
    
    def cap_nhat_chi_tiet_frequency(self, freq_sim, energy_sim, similarity_score=None, fft_info1=None, fft_info2=None):
        """Cập nhật thông tin Frequency Domain"""
        if fft_info1:
            self.label_freq_fft1.config(text=f"  📈 FFT: {fft_info1}")
        if fft_info2:
            self.label_freq_fft2.config(text=f"  FFT: {fft_info2}")
        self.label_freq_spectrum.config(text=f"Phổ tần: {freq_sim:.2f}%")
        self.label_freq_energy.config(text=f"Năng lượng: {energy_sim:.2f}%")
        # Tính overall similarity từ freq + energy
        overall = (freq_sim + energy_sim) / 2 if similarity_score is None else similarity_score
        self.label_freq_similarity.config(text=f"Tương đồng: {overall:.2f}%")
    
    def cap_nhat_thong_tin(self, kich_thuoc, num_ending, num_bifurcation):
        """Cập nhật thông tin ảnh"""
        h, w = kich_thuoc[:2]
        self.label_kich_thuoc.config(text=f"Kích thước: {w}×{h} pixels")
        
        self.label_ending.config(text=f"Kết thúc: {num_ending}")
        self.label_bifurcation.config(text=f"Phân nhánh: {num_bifurcation}")
        total = num_ending + num_bifurcation
        self.label_total.config(text=f"Tổng: {total}")
    
    def cap_nhat_ket_qua_so_khop(self, match_percentage, similarity_score):
        """Cập nhật kết quả so khớp"""
        self.label_match.config(text=f"Khớp: {match_percentage:.1f}%")
        self.label_similarity.config(text=f"Tương đồng: {similarity_score:.1f}/100")
        
        # Kiểm tra consistency: nếu Khớp thấp nhưng Tương đồng cao
        # → 2 ảnh có thể khác nhau nhưng có cơ cấu tương tự
        if match_percentage < 10 and similarity_score > 70:
            self.label_warning.config(
                text="Cảnh báo: Khớp thấp nhưng tương đồng cao - 2 ảnh có cơ cấu tương tự nhưng có thể khác nhau"
            )
        else:
            self.label_warning.config(text="")
    
    def cap_nhat_phuong_phap_so_khop(self, phương_pháp):
        """Cập nhật tiêu đề và ẩn/hiển thị labels theo phương pháp matching"""
        emoji_map = {
            'minutiae': 'MINUTIAE',
            'feature': 'FEATURE MATCHING',
            'lbp': 'LBP TEXTURE',
            'ridge': 'RIDGE ORIENTATION',
            'frequency': 'FREQUENCY DOMAIN'
        }
        
        title = emoji_map.get(phương_pháp, 'THÔNG TIN')
        
        try:
            self.lbl_details.config(text=title)
            
            # Ẩn tất cả labels
            self.label_minutiae_img1_title.grid_remove()
            self.label_ending.grid_remove()
            self.label_bifurcation.grid_remove()
            self.label_total.grid_remove()
            self.label_minutiae_img2_title.grid_remove()
            self.label_ending2.grid_remove()
            self.label_bifurcation2.grid_remove()
            self.label_total2.grid_remove()
            self.label_feature_img1_title.grid_remove()
            self.label_feature_count1.grid_remove()
            self.label_feature_img2_title.grid_remove()
            self.label_feature_count2.grid_remove()
            self.label_good_matches.grid_remove()
            self.label_lbp_img1_title.grid_remove()
            self.label_lbp_histogram1.grid_remove()
            self.label_lbp_img2_title.grid_remove()
            self.label_lbp_histogram2.grid_remove()
            self.label_lbp_distance.grid_remove()
            self.label_lbp_similarity.grid_remove()
            self.label_ridge_img1_title.grid_remove()
            self.label_ridge_orientation1.grid_remove()
            self.label_ridge_img2_title.grid_remove()
            self.label_ridge_orientation2.grid_remove()
            self.label_ridge_diff.grid_remove()
            self.label_ridge_consistency.grid_remove()
            self.label_freq_img1_title.grid_remove()
            self.label_freq_fft1.grid_remove()
            self.label_freq_img2_title.grid_remove()
            self.label_freq_fft2.grid_remove()
            self.label_freq_spectrum.grid_remove()
            self.label_freq_energy.grid_remove()
            self.label_freq_similarity.grid_remove()
            
            # Hiển thị labels tương ứng với phương pháp
            if phương_pháp == 'minutiae':
                self.label_minutiae_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_ending.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_bifurcation.grid(row=2, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_total.grid(row=3, column=0, sticky=tk.W, pady=3, padx=0)
                self.label_minutiae_img2_title.grid(row=4, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_ending2.grid(row=5, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_bifurcation2.grid(row=6, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_total2.grid(row=7, column=0, sticky=tk.W, pady=3, padx=0)
                self.notebook.tab(self.tab_minutiae_index, state='normal')
            elif phương_pháp == 'feature':
                self.label_feature_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_feature_count1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_feature_img2_title.grid(row=2, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_feature_count2.grid(row=3, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_good_matches.grid(row=4, column=0, sticky=tk.W, pady=3, padx=0)
                self.notebook.tab(self.tab_minutiae_index, state='disabled')
            elif phương_pháp == 'lbp':
                self.label_lbp_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_lbp_histogram1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_lbp_img2_title.grid(row=2, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_lbp_histogram2.grid(row=3, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_lbp_distance.grid(row=4, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_lbp_similarity.grid(row=5, column=0, sticky=tk.W, pady=3, padx=0)
                self.notebook.tab(self.tab_minutiae_index, state='disabled')
            elif phương_pháp == 'ridge':
                self.label_ridge_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_ridge_orientation1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_ridge_img2_title.grid(row=2, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_ridge_orientation2.grid(row=3, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_ridge_diff.grid(row=4, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_ridge_consistency.grid(row=5, column=0, sticky=tk.W, pady=3, padx=0)
                self.notebook.tab(self.tab_minutiae_index, state='disabled')
            elif phương_pháp == 'frequency':
                self.label_freq_img1_title.grid(row=0, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_freq_fft1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_freq_img2_title.grid(row=2, column=0, sticky=tk.W, pady=(5, 2), padx=0)
                self.label_freq_fft2.grid(row=3, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_freq_spectrum.grid(row=4, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_freq_energy.grid(row=5, column=0, sticky=tk.W, pady=2, padx=0)
                self.label_freq_similarity.grid(row=6, column=0, sticky=tk.W, pady=3, padx=0)
                self.notebook.tab(self.tab_minutiae_index, state='disabled')
        except Exception as e:
            pass
    
    def cap_nhat_thong_bao(self, tin_nhan):
        """Cập nhật thông báo thành công vào text widget"""
        self.text_thong_bao.config(state=tk.NORMAL)
        
        # Thêm dòng mới vào cuối
        if self.text_thong_bao.get("1.0", tk.END).strip():
            self.text_thong_bao.insert(tk.END, "\n")
        
        self.text_thong_bao.insert(tk.END, tin_nhan)
        
        # Scroll tới cuối
        self.text_thong_bao.see(tk.END)
        
        self.text_thong_bao.config(state=tk.DISABLED)
