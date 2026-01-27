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
        
        # Tab 1: Ảnh gốc
        tab_anh_goc = ttk.Frame(self.notebook)
        self.notebook.add(tab_anh_goc, text="📷 Ảnh gốc")
        self.canvas_anh_goc = tk.Canvas(tab_anh_goc, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_goc.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_anh_goc = None
        
        # Tab 2: Ảnh sau xử lý
        tab_anh_sau = ttk.Frame(self.notebook)
        self.notebook.add(tab_anh_sau, text="⚙️ Sau xử lý")
        self.canvas_anh_sau = tk.Canvas(tab_anh_sau, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_sau.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_anh_sau = None
        
        # Tab 3: Minutiae
        tab_anh_minutiae = ttk.Frame(self.notebook)
        self.tab_minutiae_index = self.notebook.add(tab_anh_minutiae, text="🔍 Minutiae")
        self.canvas_anh_minutiae = tk.Canvas(tab_anh_minutiae, bg="#2b2b2b", highlightthickness=0)
        self.canvas_anh_minutiae.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_anh_minutiae = None
        
        # === PHẦN 2: THÔNG TIN (Bên phải) ===
        frame_info_container = ttk.Frame(paned_window)
        paned_window.add(frame_info_container, weight=1)
        
        # Nút tải ảnh ở trên cùng
        button_frame = ttk.Frame(frame_info_container)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(button_frame, text="Tải ảnh:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        # Note: Các button sẽ được set command từ xu_ly_su_kien sau
        self.btn_anh_1 = ttk.Button(button_frame, text="📁 Ảnh 1", width=12)
        self.btn_anh_1.pack(side=tk.LEFT, padx=3)
        
        self.btn_anh_2 = ttk.Button(button_frame, text="📁 Ảnh 2", width=12)
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
        
        lbl_anh = ttk.Label(frame_anh, text="📐 THÔNG TIN ẢNH", style='Title.TLabel')
        lbl_anh.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        frame_anh_content = ttk.Frame(frame_anh, style='Card.TFrame')
        frame_anh_content.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        self.label_kich_thuoc = ttk.Label(frame_anh_content, text="📏 Kích thước: N/A", foreground='#27ae60')
        self.label_kich_thuoc.pack(anchor=tk.W, pady=3)
        
        # Thông tin chi tiết - Card style (nội dung sẽ thay đổi theo phương pháp)
        self.frame_details = ttk.Frame(self.frame_info, style='Card.TFrame')
        self.frame_details.pack(fill=tk.X, padx=8, pady=8)
        
        self.lbl_details = ttk.Label(self.frame_details, text="🔎 MINUTIAE", style='Title.TLabel')
        self.lbl_details.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        frame_details_content = ttk.Frame(self.frame_details, style='Card.TFrame')
        frame_details_content.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        # Minutiae labels
        self.label_ending = ttk.Label(frame_details_content, text="↳ Ending: 0", foreground='#3498db')
        self.label_ending.grid(row=0, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_bifurcation = ttk.Label(frame_details_content, text="↴ Bifurcation: 0", foreground='#e74c3c')
        self.label_bifurcation.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_total = ttk.Label(frame_details_content, text="✓ Tổng: 0", foreground='#f39c12')
        self.label_total.grid(row=2, column=0, sticky=tk.W, pady=3, padx=0)
        
        # Feature Matching labels
        self.label_feature_count1 = ttk.Label(frame_details_content, text="🔍 Features ảnh 1: 0", foreground='#3498db')
        self.label_feature_count1.grid(row=0, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_feature_count2 = ttk.Label(frame_details_content, text="🔍 Features ảnh 2: 0", foreground='#e74c3c')
        self.label_feature_count2.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_good_matches = ttk.Label(frame_details_content, text="✓ Good matches: 0", foreground='#f39c12')
        self.label_good_matches.grid(row=2, column=0, sticky=tk.W, pady=3, padx=0)
        
        # Harris Corners labels
        self.label_harris_corner1 = ttk.Label(frame_details_content, text="🔺 Corners ảnh 1: 0", foreground='#3498db')
        self.label_harris_corner1.grid(row=0, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_harris_corner2 = ttk.Label(frame_details_content, text="🔺 Corners ảnh 2: 0", foreground='#e74c3c')
        self.label_harris_corner2.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        
        # ORB labels
        self.label_orb_kp1 = ttk.Label(frame_details_content, text="🌀 Keypoints ảnh 1: 0", foreground='#3498db')
        self.label_orb_kp1.grid(row=0, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_orb_kp2 = ttk.Label(frame_details_content, text="🌀 Keypoints ảnh 2: 0", foreground='#e74c3c')
        self.label_orb_kp2.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_orb_matched = ttk.Label(frame_details_content, text="✓ Matched: 0", foreground='#f39c12')
        self.label_orb_matched.grid(row=2, column=0, sticky=tk.W, pady=3, padx=0)
        
        # LBP labels
        self.label_lbp_distance = ttk.Label(frame_details_content, text="📊 Chi-square distance: 0.0000", foreground='#3498db')
        self.label_lbp_distance.grid(row=0, column=0, sticky=tk.W, pady=2, padx=0)
        
        # Ridge Orientation labels
        self.label_ridge_diff = ttk.Label(frame_details_content, text="〰️ Mean angle diff: 0.00°", foreground='#3498db')
        self.label_ridge_diff.grid(row=0, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_ridge_cons1 = ttk.Label(frame_details_content, text="〰️ Consistency 1: 0.0000", foreground='#e74c3c')
        self.label_ridge_cons1.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_ridge_cons2 = ttk.Label(frame_details_content, text="〰️ Consistency 2: 0.0000", foreground='#f39c12')
        self.label_ridge_cons2.grid(row=2, column=0, sticky=tk.W, pady=3, padx=0)
        
        # Frequency Domain labels
        self.label_freq_sim = ttk.Label(frame_details_content, text="📈 Freq similarity: 0.00", foreground='#3498db')
        self.label_freq_sim.grid(row=0, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_energy_sim = ttk.Label(frame_details_content, text="📈 Energy similarity: 0.00", foreground='#e74c3c')
        self.label_energy_sim.grid(row=1, column=0, sticky=tk.W, pady=2, padx=0)
        
        self.label_ridge_sim = ttk.Label(frame_details_content, text="📈 Ridge similarity: 0.00", foreground='#f39c12')
        self.label_ridge_sim.grid(row=2, column=0, sticky=tk.W, pady=3, padx=0)
        
        # Thông tin so khớp - Card style
        frame_match = ttk.Frame(self.frame_info, style='Card.TFrame')
        frame_match.pack(fill=tk.X, padx=8, pady=8)
        
        lbl_match = ttk.Label(frame_match, text="⚖️ KẾT QUẢ SO KHỚP", style='Title.TLabel')
        lbl_match.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        frame_match_content = ttk.Frame(frame_match, style='Card.TFrame')
        frame_match_content.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        self.label_match = ttk.Label(frame_match_content, text="🎯 Match: N/A", foreground='#9b59b6')
        self.label_match.pack(anchor=tk.W, pady=2)
        
        self.label_similarity = ttk.Label(frame_match_content, text="📊 Tương đồng: N/A", foreground='#1abc9c')
        self.label_similarity.pack(anchor=tk.W, pady=3)
        
        # Thông báo thành công - Card style
        frame_notification = ttk.Frame(self.frame_info, style='Card.TFrame')
        frame_notification.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        lbl_thong_bao = ttk.Label(frame_notification, text="✅ THÔNG BÁO", style='Title.TLabel')
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
    
    def hien_thi_anh_goc(self, anh):
        """Hiển thị ảnh gốc"""
        self._hien_thi_anh_len_canvas(anh, self.canvas_anh_goc)
    
    def hien_thi_anh_sau_xu_ly(self, anh):
        """Hiển thị ảnh sau xử lý"""
        self._hien_thi_anh_len_canvas(anh, self.canvas_anh_sau)
    
    def hien_thi_anh_after_xu_ly(self, anh):
        """Hiển thị ảnh minutiae"""
        self._hien_thi_anh_len_canvas(anh, self.canvas_anh_minutiae)
    
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
    
    def cap_nhat_chi_tiet_minutiae(self, ending_count, bifurcation_count, total_count):
        """Cập nhật thông tin Minutiae Matching"""
        self.label_ending.config(text=f"📍 Ending: {ending_count}")
        self.label_bifurcation.config(text=f"🔀 Bifurcation: {bifurcation_count}")
        self.label_total.config(text=f"📊 Total: {total_count}")
    
    def cap_nhat_chi_tiet_feature(self, feature_count1, feature_count2, good_matches):
        """Cập nhật thông tin Feature Matching"""
        self.label_feature_count1.config(text=f"🔍 Features ảnh 1: {feature_count1}")
        self.label_feature_count2.config(text=f"🔍 Features ảnh 2: {feature_count2}")
        self.label_good_matches.config(text=f"✓ Good matches: {good_matches}")
    
    def cap_nhat_chi_tiet_lbp(self, chi_square_distance):
        """Cập nhật thông tin LBP Texture"""
        self.label_lbp_distance.config(text=f"📊 Chi-square distance: {chi_square_distance:.4f}")
    
    def cap_nhat_chi_tiet_ridge(self, mean_angle_diff, consistency1, consistency2):
        """Cập nhật thông tin Ridge Orientation"""
        self.label_ridge_diff.config(text=f"〰️ Mean angle diff: {mean_angle_diff:.2f}°")
        self.label_ridge_cons1.config(text=f"〰️ Consistency 1: {consistency1:.4f}")
        self.label_ridge_cons2.config(text=f"〰️ Consistency 2: {consistency2:.4f}")
    
    def cap_nhat_chi_tiet_frequency(self, freq_sim, energy_sim, ridge_sim):
        """Cập nhật thông tin Frequency Domain"""
        self.label_freq_sim.config(text=f"📈 Freq similarity: {freq_sim:.2f}")
        self.label_energy_sim.config(text=f"📈 Energy similarity: {energy_sim:.2f}")
        self.label_ridge_sim.config(text=f"📈 Ridge similarity: {ridge_sim:.2f}")
    
    def cap_nhat_thong_tin(self, kich_thuoc, num_ending, num_bifurcation):
        """Cập nhật thông tin ảnh"""
        h, w = kich_thuoc[:2]
        self.label_kich_thuoc.config(text=f"📏 Kích thước: {w}×{h} pixels")
        
        self.label_ending.config(text=f"↳ Ending: {num_ending}")
        self.label_bifurcation.config(text=f"↴ Bifurcation: {num_bifurcation}")
        total = num_ending + num_bifurcation
        self.label_total.config(text=f"✓ Tổng: {total}")
    
    def cap_nhat_ket_qua_so_khop(self, match_percentage, similarity_score):
        """Cập nhật kết quả so khớp"""
        self.label_match.config(text=f"🎯 Match: {match_percentage:.1f}%")
        self.label_similarity.config(text=f"📊 Tương đồng: {similarity_score:.1f}/100")
    
    def cap_nhat_phuong_phap_so_khop(self, phương_pháp):
        """Cập nhật tiêu đề và ẩn/hiển thị labels theo phương pháp matching"""
        emoji_map = {
            'minutiae': '🔎 MINUTIAE',
            'feature': '🎯 FEATURE MATCHING',
            'lbp': '📊 LBP TEXTURE',
            'ridge': '〰️ RIDGE ORIENTATION',
            'frequency': '📈 FREQUENCY DOMAIN'
        }
        
        title = emoji_map.get(phương_pháp, '🔍 THÔNG TIN')
        
        try:
            self.lbl_details.config(text=title)
            
            # Ẩn tất cả labels
            self.label_ending.grid_remove()
            self.label_bifurcation.grid_remove()
            self.label_total.grid_remove()
            self.label_feature_count1.grid_remove()
            self.label_feature_count2.grid_remove()
            self.label_good_matches.grid_remove()
            self.label_lbp_distance.grid_remove()
            self.label_ridge_diff.grid_remove()
            self.label_ridge_cons1.grid_remove()
            self.label_ridge_cons2.grid_remove()
            self.label_freq_sim.grid_remove()
            self.label_energy_sim.grid_remove()
            self.label_ridge_sim.grid_remove()
            
            # Hiển thị labels tương ứng với phương pháp
            if phương_pháp == 'minutiae':
                self.label_ending.grid()
                self.label_bifurcation.grid()
                self.label_total.grid()
                self.notebook.tab(self.tab_minutiae_index, state='normal')
            elif phương_pháp == 'feature':
                self.label_feature_count1.grid()
                self.label_feature_count2.grid()
                self.label_good_matches.grid()
                self.notebook.tab(self.tab_minutiae_index, state='disabled')
            elif phương_pháp == 'lbp':
                self.label_lbp_distance.grid()
                self.notebook.tab(self.tab_minutiae_index, state='disabled')
            elif phương_pháp == 'ridge':
                self.label_ridge_diff.grid()
                self.label_ridge_cons1.grid()
                self.label_ridge_cons2.grid()
                self.notebook.tab(self.tab_minutiae_index, state='disabled')
            elif phương_pháp == 'frequency':
                self.label_freq_sim.grid()
                self.label_energy_sim.grid()
                self.label_ridge_sim.grid()
                self.notebook.tab(self.tab_minutiae_index, state='disabled')
        except Exception as e:
            pass
    
    def cap_nhat_thong_bao(self, tin_nhan):
        """Cập nhật thông báo thành công vào text widget"""
        self.text_thong_bao.config(state=tk.NORMAL)
        
        # Thêm dòng mới vào cuối
        if self.text_thong_bao.get("1.0", tk.END).strip():
            self.text_thong_bao.insert(tk.END, "\n")
        
        self.text_thong_bao.insert(tk.END, f"✓ {tin_nhan}")
        
        # Scroll tới cuối
        self.text_thong_bao.see(tk.END)
        
        self.text_thong_bao.config(state=tk.DISABLED)
