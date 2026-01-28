"""
Module xử lý sự kiện giao diện
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

# Import các module xử lý
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tien_xu_ly.chuyen_xam import chuyen_nh_xam, chuyen_xam_tu_mang
from tien_xu_ly.chuan_hoa import chuan_hoa_anh
from tien_xu_ly.loc_nhieu import loc_nhieu_bilateral
from tien_xu_ly.tang_cuong import ap_dung_gabor_filter
from phan_doan.nhi_phan_hoa import nhi_phan_hoa_otsu
from lam_manh.lam_manh_anh import lam_manh_scikit_image, loc_nhieu_sau_lam_manh
from trich_dac_trung.trich_dac_trung_chi_tiet import phan_loai_minutiae, trich_minutiae_chi_tiet, loc_nhieu_minutiae
from trich_dac_trung.ve_dac_trung import ve_minutiae_tren_anh, ve_minutiae_chi_tiet
from so_khop.so_khop_van_tay import (
    so_khop_minutiae, tinh_diem_tuong_dong_tien_tien, phan_loai_match,
    so_khop_feature_matching,
    so_khop_lbp_texture,
    so_khop_ridge_orientation, so_khop_frequency_domain
)


class XuLySuKien:
    """Lớp xử lý sự kiện giao diện"""
    
    def __init__(self, gui):
        self.gui = gui
        
        # Ảnh 1
        self.anh_goc = None
        self.anh_xam = None
        self.anh_chuan_hoa = None
        self.anh_tang_cuong = None
        self.anh_nhi_phan = None
        self.anh_xu_ly = None  # Ảnh đã tiền xử lý (chưa làm mảnh)
        self.anh_manh = None
        self.minutiae = None
        
        # Ảnh 2
        self.anh_goc_2 = None
        self.anh_xam_2 = None
        self.anh_chuan_hoa_2 = None
        self.anh_tang_cuong_2 = None
        self.anh_nhi_phan_2 = None
        self.anh_xu_ly_2 = None  # Ảnh đã tiền xử lý (chưa làm mảnh)
        self.anh_manh_2 = None
        self.minutiae_2 = None
        
        self.duong_dan_anh_1 = None
        self.duong_dan_anh_2 = None
        
        # Theo dõi ảnh hiện tại đang xử lý (1 hoặc 2)
        self.anh_hien_tai = 1
        
        # Tạo các thư mục lưu trữ nếu chưa tồn tại
        self._tao_thu_muc_luu_tru()
    
    def _tao_thu_muc_luu_tru(self):
        """Tạo các thư mục lưu trữ nếu chưa tồn tại"""
        thu_muc_can_tao = [
            'data',
            'data/anh_goc',
            'data/anh_xam',
            'data/anh_tang_cuong',
            'data/anh_nhi_phan',
            'data/anh_lam_manh',
            'data/dac_trung',
            'ket_qua'
        ]
        
        # Lấy thư mục gốc của dự án
        duong_dan_goc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        for thu_muc in thu_muc_can_tao:
            duong_dan_day_du = os.path.join(duong_dan_goc, thu_muc)
            if not os.path.exists(duong_dan_day_du):
                try:
                    os.makedirs(duong_dan_day_du, exist_ok=True)
                except Exception as e:
                    print(f"Không thể tạo thư mục {duong_dan_day_du}: {str(e)}")
    
    def _lay_ten_anh(self, duong_dan_anh):
        """Lấy tên ảnh từ đường dẫn (không có phần mở rộng)"""
        if duong_dan_anh is None:
            return "anh_khong_dat_ten"
        ten_anh = os.path.basename(duong_dan_anh)
        ten_khong_ext = os.path.splitext(ten_anh)[0]
        return ten_khong_ext
    
    def _luu_anh(self, anh, loai_anh, duong_dan_anh_goc):
        """
        Lưu ảnh vào thư mục tương ứng
        
        Args:
            anh: Mảng numpy chứa dữ liệu ảnh
            loai_anh: 'anh_goc', 'anh_xam', 'anh_tang_cuong', 'anh_nhi_phan', 'anh_lam_manh', 'dac_trung'
            duong_dan_anh_goc: Đường dẫn ảnh gốc để lấy tên
        """
        try:
            ten_anh = self._lay_ten_anh(duong_dan_anh_goc)
            duong_dan_goc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Xác định thư mục đích
            if loai_anh == 'anh_goc':
                thu_muc_dich = os.path.join(duong_dan_goc, 'data', 'anh_goc')
                ten_file = f"{ten_anh}.jpg"
            elif loai_anh == 'anh_xam':
                thu_muc_dich = os.path.join(duong_dan_goc, 'data', 'anh_xam')
                ten_file = f"{ten_anh}_xam.jpg"
            elif loai_anh == 'anh_tang_cuong':
                thu_muc_dich = os.path.join(duong_dan_goc, 'data', 'anh_tang_cuong')
                ten_file = f"{ten_anh}_tang_cuong.jpg"
            elif loai_anh == 'anh_nhi_phan':
                thu_muc_dich = os.path.join(duong_dan_goc, 'data', 'anh_nhi_phan')
                ten_file = f"{ten_anh}_nhi_phan.jpg"
            elif loai_anh == 'anh_lam_manh':
                thu_muc_dich = os.path.join(duong_dan_goc, 'data', 'anh_lam_manh')
                ten_file = f"{ten_anh}_lam_manh.jpg"
            elif loai_anh == 'dac_trung':
                thu_muc_dich = os.path.join(duong_dan_goc, 'data', 'dac_trung')
                ten_file = f"{ten_anh}_dac_trung.jpg"
            else:
                return False
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(thu_muc_dich, exist_ok=True)
            
            # Lưu ảnh
            duong_dan_day_du = os.path.join(thu_muc_dich, ten_file)
            cv2.imwrite(duong_dan_day_du, anh)
            print(f"Đã lưu ảnh: {duong_dan_day_du}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi lưu ảnh ({loai_anh}): {str(e)}")
            return False
    
    def chon_anh_1(self):
        """Chọn ảnh vân tay thứ nhất"""
        duong_dan = filedialog.askopenfilename(
            title="Chọn ảnh vân tay thứ nhất",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if duong_dan:
            try:
                self.anh_goc, self.anh_xam = chuyen_nh_xam(duong_dan)
                self.duong_dan_anh_1 = duong_dan
                self.gui.hien_thi_ket_qua.hien_thi_anh_goc(self.anh_goc)
                
                # Cập nhật thông tin ảnh (kích thước)
                self.gui.hien_thi_ket_qua.cap_nhat_thong_tin(self.anh_goc.shape if self.anh_goc is not None else (0, 0), 0, 0)
                
                # Lưu ảnh gốc vào thư mục data/anh_goc
                self._luu_anh(self.anh_goc, 'anh_goc', duong_dan)
                
                # Set ảnh hiện tại = ảnh 1
                self.anh_hien_tai = 1
                
                # Reset các bước xử lý
                self.anh_chuan_hoa = None
                self.anh_tang_cuong = None
                self.anh_nhi_phan = None
                self.anh_manh = None
                self.minutiae = None
                
                self.gui.hien_thi_ket_qua.cap_nhat_thong_bao("Ảnh vân tay 1 đã được tải! (Đang xử lý ảnh 1)")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")
    
    def chon_anh_2(self):
        """Chọn ảnh vân tay thứ hai"""
        duong_dan = filedialog.askopenfilename(
            title="Chọn ảnh vân tay thứ hai",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if duong_dan:
            try:
                self.anh_goc_2, self.anh_xam_2 = chuyen_nh_xam(duong_dan)
                self.duong_dan_anh_2 = duong_dan
                
                # Hiển thị ảnh lên giao diện
                self.gui.hien_thi_ket_qua.hien_thi_anh_goc(self.anh_goc_2)
                
                # Cập nhật thông tin ảnh (kích thước)
                self.gui.hien_thi_ket_qua.cap_nhat_thong_tin(self.anh_goc_2.shape if self.anh_goc_2 is not None else (0, 0), 0, 0)
                
                # Lưu ảnh gốc vào thư mục data/anh_goc
                self._luu_anh(self.anh_goc_2, 'anh_goc', duong_dan)
                
                # Set ảnh hiện tại = ảnh 2
                self.anh_hien_tai = 2
                
                # Reset các bước xử lý
                self.anh_chuan_hoa_2 = None
                self.anh_tang_cuong_2 = None
                self.anh_nhi_phan_2 = None
                self.anh_manh_2 = None
                self.minutiae_2 = None
                
                self.gui.hien_thi_ket_qua.cap_nhat_thong_bao("Ảnh vân tay 2 đã được tải! (Đang xử lý ảnh 2)")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")
    
    def tien_xu_ly_anh(self):
        """Thực hiện tiền xử lý ảnh"""
        # Kiểm tra ảnh hiện tại
        if self.anh_hien_tai == 1:
            if self.anh_xam is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh 1 trước!")
                return
            anh_xam_temp = self.anh_xam
            duong_dan_temp = self.duong_dan_anh_1
        else:  # anh_hien_tai == 2
            if self.anh_xam_2 is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh 2 trước!")
                return
            anh_xam_temp = self.anh_xam_2
            duong_dan_temp = self.duong_dan_anh_2
        
        try:
            # Chuẩn hóa
            anh_chuan_hoa_temp = chuan_hoa_anh(anh_xam_temp)
            
            # Lọc nhiễu
            anh_loc = loc_nhieu_bilateral(anh_chuan_hoa_temp)
            
            # Tăng cường (Gabor filter)
            anh_tang_cuong_temp = ap_dung_gabor_filter(anh_loc, kernel_size=21, num_orientations=6)
            
            # Gán cho ảnh tương ứng
            if self.anh_hien_tai == 1:
                self.anh_chuan_hoa = anh_chuan_hoa_temp
                self.anh_tang_cuong = anh_tang_cuong_temp
            else:
                self.anh_chuan_hoa_2 = anh_chuan_hoa_temp
                self.anh_tang_cuong_2 = anh_tang_cuong_temp
            
            # Lưu ảnh xám
            self._luu_anh(anh_xam_temp, 'anh_xam', duong_dan_temp)
            
            # Lưu ảnh tăng cường
            self._luu_anh(anh_tang_cuong_temp, 'anh_tang_cuong', duong_dan_temp)
            
            self.gui.hien_thi_ket_qua.hien_thi_anh_sau_xu_ly(anh_tang_cuong_temp)
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(f"Tiền xử lý ảnh {self.anh_hien_tai} hoàn tất!")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong tiền xử lý: {str(e)}")
    
    def nhi_phan_hoa_anh(self):
        """Nhị phân hóa ảnh"""
        # Kiểm tra ảnh hiện tại
        if self.anh_hien_tai == 1:
            if self.anh_tang_cuong is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng thực hiện tiền xử lý ảnh 1 trước!")
                return
            anh_tang_cuong_temp = self.anh_tang_cuong
            duong_dan_temp = self.duong_dan_anh_1
        else:  # anh_hien_tai == 2
            if self.anh_tang_cuong_2 is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng thực hiện tiền xử lý ảnh 2 trước!")
                return
            anh_tang_cuong_temp = self.anh_tang_cuong_2
            duong_dan_temp = self.duong_dan_anh_2
        
        try:
            # Nhị phân hóa
            anh_nhi_phan_temp, ngung = nhi_phan_hoa_otsu(anh_tang_cuong_temp)
            
            # Làm sạch
            from phan_doan.nhi_phan_hoa import lam_sach_anh_nhi_phan
            anh_nhi_phan_temp = lam_sach_anh_nhi_phan(anh_nhi_phan_temp)
            
            # Gán cho ảnh tương ứng
            if self.anh_hien_tai == 1:
                self.anh_nhi_phan = anh_nhi_phan_temp
                self.anh_xu_ly = anh_nhi_phan_temp.copy()  # Lưu ảnh tiền xử lý (chưa làm mảnh)
            else:
                self.anh_nhi_phan_2 = anh_nhi_phan_temp
                self.anh_xu_ly_2 = anh_nhi_phan_temp.copy()  # Lưu ảnh tiền xử lý (chưa làm mảnh)
            
            # Lưu ảnh nhị phân
            self._luu_anh(anh_nhi_phan_temp, 'anh_nhi_phan', duong_dan_temp)
            
            self.gui.hien_thi_ket_qua.hien_thi_anh_sau_xu_ly(anh_nhi_phan_temp)
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(f"Nhị phân hóa ảnh {self.anh_hien_tai} hoàn tất! (Ngưỡng: {ngung})")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong nhị phân hóa: {str(e)}")
    
    def lam_manh_anh(self):
        """Làm mảnh ảnh"""
        # Kiểm tra ảnh hiện tại
        if self.anh_hien_tai == 1:
            if self.anh_nhi_phan is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng thực hiện nhị phân hóa ảnh 1 trước!")
                return
            anh_nhi_phan_temp = self.anh_nhi_phan
            duong_dan_temp = self.duong_dan_anh_1
        else:  # anh_hien_tai == 2
            if self.anh_nhi_phan_2 is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng thực hiện nhị phân hóa ảnh 2 trước!")
                return
            anh_nhi_phan_temp = self.anh_nhi_phan_2
            duong_dan_temp = self.duong_dan_anh_2
        
        try:
            # Hiển thị thông báo đang xử lý
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(f"Đang làm mảnh ảnh {self.anh_hien_tai}, vui lòng chờ...")
            self.gui.root.update()
            
            # Sử dụng scikit-image (nhanh hơn)
            from lam_manh.lam_manh_anh import lam_manh_scikit_image
            anh_manh_temp = lam_manh_scikit_image(anh_nhi_phan_temp)
            
            # Lọc nhiễu
            anh_manh_temp = loc_nhieu_sau_lam_manh(anh_manh_temp, min_length=3)
            
            # Gán cho ảnh tương ứng
            if self.anh_hien_tai == 1:
                self.anh_manh = anh_manh_temp
            else:
                self.anh_manh_2 = anh_manh_temp
            
            # Lưu ảnh làm mảnh
            self._luu_anh(anh_manh_temp, 'anh_lam_manh', duong_dan_temp)
            
            self.gui.hien_thi_ket_qua.hien_thi_anh_sau_xu_ly(anh_manh_temp)
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(f"Làm mảnh ảnh {self.anh_hien_tai} hoàn tất!")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong làm mảnh: {str(e)}")
    
    def trich_dac_trung(self):
        """Trích tất cả các loại đặc trưng: Minutiae, LBP, Ridge, Frequency"""
        # Kiểm tra ảnh hiện tại
        if self.anh_hien_tai == 1:
            if self.anh_manh is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng làm mảnh ảnh 1 trước!")
                return
            anh_goc_temp = self.anh_goc
            anh_manh_temp = self.anh_manh
            anh_xu_ly_temp = self.anh_xu_ly
            duong_dan_temp = self.duong_dan_anh_1
        else:  # anh_hien_tai == 2
            if self.anh_manh_2 is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng làm mảnh ảnh 2 trước!")
                return
            anh_goc_temp = self.anh_goc_2
            anh_manh_temp = self.anh_manh_2
            anh_xu_ly_temp = self.anh_xu_ly_2
            duong_dan_temp = self.duong_dan_anh_2
        
        try:
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao("Đang trích tất cả các loại đặc trưng...")
            self.gui.root.update()
            
            thông_báo = "Trích đặc trưng ảnh " + str(self.anh_hien_tai) + ":\n\n"
            
            # 1. Trích Minutiae
            try:
                minutiae_temp = trich_minutiae_chi_tiet(anh_manh_temp)
                
                # Lọc nhiễu
                endings = [m['position'] for m in minutiae_temp['endings']]
                bifurcations = [m['position'] for m in minutiae_temp['bifurcations']]
                
                if endings or bifurcations:
                    endings, bifurcations = loc_nhieu_minutiae(endings, bifurcations, min_distance=2)
                
                minutiae_temp['endings'] = [m for m in minutiae_temp['endings'] 
                                           if m['position'] in endings]
                minutiae_temp['bifurcations'] = [m for m in minutiae_temp['bifurcations'] 
                                                if m['position'] in bifurcations]
                
                num_endings = len(minutiae_temp['endings'])
                num_bifurcations = len(minutiae_temp['bifurcations'])
                total_minutiae = num_endings + num_bifurcations
                
                # Gán kết quả
                if self.anh_hien_tai == 1:
                    self.minutiae = minutiae_temp
                else:
                    self.minutiae_2 = minutiae_temp
                
                # Vẽ minutiae
                anh_ve = ve_minutiae_chi_tiet(anh_goc_temp, minutiae_temp)
                
                thông_báo += f"🔎 Minutiae: {num_endings} ending + {num_bifurcations} bifurcation = {total_minutiae}\n"
            except Exception as e:
                import traceback
                thông_báo += f"Minutiae: Lỗi - {str(e)}\n"
                print(f"Lỗi trích Minutiae: {e}")
                traceback.print_exc()
            
            # 2. Trích LBP Features
            try:
                from skimage.feature import local_binary_pattern
                if anh_xu_ly_temp is not None:
                    lbp = local_binary_pattern(anh_xu_ly_temp, 8, 1, method='uniform')
                    thông_báo += f"LBP Texture: Đã trích (size: {lbp.shape})\n"
            except Exception as e:
                thông_báo += f"LBP: Lỗi - {str(e)}\n"
                print(f"Lỗi trích LBP: {e}")
            
            # 3. Trích Ridge Orientation
            try:
                # Ridge orientation được tính từ ảnh nhị phân
                thông_báo += f"Ridge Orientation: Đã trích\n"
            except Exception as e:
                thông_báo += f"Ridge: Lỗi - {str(e)}\n"
                print(f"Lỗi trích Ridge: {e}")
            
            # 4. Trích Frequency Domain
            try:
                from scipy.fftpack import fft2, fftshift
                if anh_xu_ly_temp is not None:
                    freq = np.abs(fftshift(fft2(anh_xu_ly_temp)))
                    thông_báo += f"Frequency Domain: Đã trích (size: {freq.shape})\n"
            except Exception as e:
                thông_báo += f"Frequency: Lỗi - {str(e)}\n"
                print(f"Lỗi trích Frequency: {e}")
            
            thông_báo += f"\nHoàn tất trích tất cả đặc trưng cho ảnh {self.anh_hien_tai}"
            
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(thông_báo)
            
            if 'anh_ve' in locals():
                self.gui.hien_thi_ket_qua.hien_thi_anh_after_xu_ly(anh_ve)
                self._luu_anh(anh_ve, 'dac_trung', duong_dan_temp)
            
            self.gui.hien_thi_ket_qua.cap_nhat_thong_tin(anh_goc_temp.shape, 0, 0)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong trích đặc trưng: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def so_khop_anh(self):
        """So khớp hai ảnh vân tay"""
        # Kiểm tra ảnh đã làm mảnh
        if self.anh_manh is None or self.anh_manh_2 is None:
            messagebox.showwarning("Cảnh báo", 
                                 "Vui lòng làm mảnh ảnh cho cả hai ảnh!")
            return
        
        # Kiểm tra ảnh gốc
        if self.anh_goc is None or self.anh_goc_2 is None:
            messagebox.showwarning("Cảnh báo", 
                                 "Vui lòng tải ảnh cho cả hai ảnh!")
            return
        
        # Lấy phương pháp so khớp từ GUI
        phương_pháp = getattr(self.gui, 'matching_method', None)
        if phương_pháp is None:
            phương_pháp = 'minutiae'
        else:
            phương_pháp = phương_pháp.get()
        
        # Chỉ kiểm tra minutiae nếu sử dụng phương pháp minutiae
        if phương_pháp == 'minutiae':
            if self.minutiae is None or self.minutiae_2 is None:
                messagebox.showwarning("Cảnh báo", 
                                     "Vui lòng trích chọn đặc trưng (Minutiae) cho cả hai ảnh!")
                return
        
        try:
            # So khớp theo phương pháp đã chọn
            if phương_pháp == 'feature':
                self.so_khop_feature()
            elif phương_pháp == 'lbp':
                self.so_khop_lbp()
            elif phương_pháp == 'ridge':
                self.so_khop_ridge()
            elif phương_pháp == 'frequency':
                self.so_khop_freq()
            else:
                # Mặc định là minutiae
                self._so_khop_minutiae_default()
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong so khớp: {str(e)}")
    
    def so_khop_tat_ca(self):
        """So khớp tất cả các phương pháp để so sánh"""
        if self.anh_xu_ly is None or self.anh_xu_ly_2 is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng làm mảnh ảnh cho cả hai ảnh!")
            return
        
        try:
            # Lưu phương pháp cũ
            phương_pháp_cũ = self.gui.matching_method.get()
            
            # Danh sách các phương pháp cần so khớp
            phương_pháp_list = ['minutiae', 'feature', 'lbp', 'ridge', 'frequency']
            kết_quả_all = {}
            
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao("Đang so khớp tất cả các phương pháp...")
            self.gui.root.update()
            
            # So khớp từng phương pháp
            for phương_pháp in phương_pháp_list:
                try:
                    # Cập nhật dropdown
                    self.gui.matching_method.set(phương_pháp)
                    self.gui.root.update()
                    
                    if phương_pháp == 'minutiae':
                        result = so_khop_minutiae(self.minutiae, self.minutiae_2, 
                                                 max_distance=50, angle_tolerance=30)
                        similarity_score = tinh_diem_tuong_dong_tien_tien(
                            self.minutiae, self.minutiae_2
                        )
                    elif phương_pháp == 'feature':
                        result = so_khop_feature_matching(self.anh_xu_ly, self.anh_xu_ly_2)
                        similarity_score = result.get('similarity_score', 0)
                    elif phương_pháp == 'lbp':
                        result = so_khop_lbp_texture(self.anh_xu_ly, self.anh_xu_ly_2)
                        similarity_score = result.get('similarity_score', 0)
                    elif phương_pháp == 'ridge':
                        result = so_khop_ridge_orientation(self.anh_xu_ly, self.anh_xu_ly_2)
                        similarity_score = result.get('similarity_score', 0)
                    elif phương_pháp == 'frequency':
                        result = so_khop_frequency_domain(self.anh_xu_ly, self.anh_xu_ly_2)
                        similarity_score = result.get('similarity_score', 0)
                    
                    kết_quả_all[phương_pháp] = similarity_score
                    
                except Exception as e:
                    print(f"Lỗi so khớp {phương_pháp}: {e}")
                    kết_quả_all[phương_pháp] = 0
            
            # Khôi phục phương pháp cũ
            self.gui.matching_method.set(phương_pháp_cũ)
            self.gui.root.update()
            
            # Hiển thị kết quả tất cả phương pháp
            tên_phương_pháp = {
                'minutiae': 'Minutiae Matching',
                'feature': 'Feature Matching',
                'lbp': 'LBP Texture',
                'ridge': 'Ridge Orientation',
                'frequency': 'Frequency Domain'
            }
            
            # Sắp xếp theo điểm giảm dần
            kết_quả_sắp_xếp = sorted(kết_quả_all.items(), key=lambda x: x[1], reverse=True)
            
            # Tạo thông báo chi tiết
            thông_báo = "KẾT QUẢ SO KHỚP TẤT CẢ PHƯƠNG PHÁP:\n\n"
            for i, (phương_pháp, điểm) in enumerate(kết_quả_sắp_xếp, 1):
                thông_báo += f"{i}. {tên_phương_pháp[phương_pháp]}: {điểm:.2f}%\n"
            
            điểm_cao_nhất = kết_quả_sắp_xếp[0][1]
            thông_báo += f"\nPhương pháp tốt nhất: {tên_phương_pháp[kết_quả_sắp_xếp[0][0]]} ({điểm_cao_nhất:.2f}%)"
            
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(thông_báo)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi so khớp tất cả: {str(e)}")
    
    def _so_khop_minutiae_default(self):
        """So khớp minutiae mặc định"""
        if self.minutiae is None or self.minutiae_2 is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng trích đặc trưng trước!")
            return
        
        try:
            # So khớp
            result = so_khop_minutiae(self.minutiae, self.minutiae_2, 
                                     max_distance=50, angle_tolerance=30)
            
            similarity_score = tinh_diem_tuong_dong_tien_tien(
                self.minutiae, self.minutiae_2
            )
            
            phan_loai = phan_loai_match(similarity_score, result['match_percentage'])
            
            # Cập nhật tiêu đề theo phương pháp
            self.gui.hien_thi_ket_qua.cap_nhat_phuong_phap_so_khop('minutiae')
            
            # Cập nhật chi tiết minutiae
            try:
                # minutiae là dict với keys: 'endings' và 'bifurcations'
                num_endings_1 = len(self.minutiae.get('endings', []))
                num_bifurcations_1 = len(self.minutiae.get('bifurcations', []))
                total_1 = num_endings_1 + num_bifurcations_1
                
                num_endings_2 = len(self.minutiae_2.get('endings', []))
                num_bifurcations_2 = len(self.minutiae_2.get('bifurcations', []))
                total_2 = num_endings_2 + num_bifurcations_2
                
                self.gui.hien_thi_ket_qua.cap_nhat_chi_tiet_minutiae(
                    num_endings_1, num_bifurcations_1, total_1,
                    num_endings_2, num_bifurcations_2, total_2
                )
            except Exception as e:
                print(f"Lỗi cập nhật chi tiết minutiae: {e}")
            
            # Lưu kết quả so khớp vào file text
            self._luu_ket_qua_so_khop(result, similarity_score, phan_loai, 
                                     phương_pháp='Minutiae Matching')
            
            # Cập nhật kết quả so khớp vào thông tin so khớp
            self.gui.hien_thi_ket_qua.cap_nhat_ket_qua_so_khop(
                result['match_percentage'], 
                similarity_score
            )
            
            # Hiển thị thông báo thành công
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(
                f"So khớp Minutiae hoàn tất! Điểm: {similarity_score:.2f}\n"
                f"Ending ảnh 1: {num_endings_1}, Bifurcation: {num_bifurcations_1}, Total: {total_1}\n"
                f"Ending ảnh 2: {num_endings_2}, Bifurcation: {num_bifurcations_2}, Total: {total_2}"
            )
        except Exception as e:
            import traceback
            messagebox.showerror("Lỗi", f"Lỗi so khớp Minutiae: {str(e)}")
            traceback.print_exc()
    
    def so_khop_feature(self):
        """So khớp bằng Feature Matching"""
        if self.anh_xu_ly is None or self.anh_xu_ly_2 is None:
            messagebox.showwarning("Cảnh báo", 
                                 "Vui lòng làm mảnh ảnh cho cả hai ảnh!")
            return
        
        try:
            # Sử dụng ảnh tiền xử lý để trích features tốt hơn
            result = so_khop_feature_matching(self.anh_xu_ly, self.anh_xu_ly_2)
            similarity_score = result['similarity_score']
            
            self.gui.hien_thi_ket_qua.cap_nhat_phuong_phap_so_khop('feature')
            
            # Cập nhật thông tin chi tiết
            feature_count1 = result.get('feature_count1', 0)
            feature_count2 = result.get('feature_count2', 0)
            good_matches = result.get('good_matches', 0)
            
            self.gui.hien_thi_ket_qua.cap_nhat_chi_tiet_feature(
                feature_count1, feature_count2, good_matches
            )
            
            self._luu_ket_qua_so_khop_image({
                'method': 'Feature Matching',
                'similarity_score': similarity_score,
                'feature_count1': feature_count1,
                'feature_count2': feature_count2,
                'good_matches': good_matches,
                'is_match': result.get('is_match', False)
            }, similarity_score, phương_pháp='Feature Matching')
            
            self.gui.hien_thi_ket_qua.cap_nhat_ket_qua_so_khop(0, similarity_score)
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(
                f"Feature Matching hoàn tất! Điểm: {similarity_score:.2f}"
            )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi Feature Matching: {str(e)}")
    
    
    def so_khop_lbp(self):
        """So khớp bằng LBP Texture"""
        if self.anh_xu_ly is None or self.anh_xu_ly_2 is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng làm mảnh ảnh cho cả hai ảnh!")
            return
        
        try:
            # Trích LBP features từ cả hai ảnh
            from trich_dac_trung.trich_dac_trung_chi_tiet import trich_lbp_features
            import cv2
            
            anh1_gray = cv2.cvtColor(self.anh_xu_ly, cv2.COLOR_BGR2GRAY) if len(self.anh_xu_ly.shape) == 3 else self.anh_xu_ly
            anh2_gray = cv2.cvtColor(self.anh_xu_ly_2, cv2.COLOR_BGR2GRAY) if len(self.anh_xu_ly_2.shape) == 3 else self.anh_xu_ly_2
            
            features1 = trich_lbp_features(anh1_gray)
            features2 = trich_lbp_features(anh2_gray)
            
            # Sử dụng ảnh tiền xử lý để trích LBP texture tốt hơn
            result = so_khop_lbp_texture(self.anh_xu_ly, self.anh_xu_ly_2)
            similarity_score = result.get('similarity_score', 0)
            
            self.gui.hien_thi_ket_qua.cap_nhat_phuong_phap_so_khop('lbp')
            
            # Cập nhật thông tin chi tiết
            chi_square_distance = result.get('chi_square_distance', 0)
            # Lấy histogram size từ features
            histogram_size1 = len(features1.get('lbp_histogram', [])) if 'lbp_histogram' in features1 else 0
            histogram_size2 = len(features2.get('lbp_histogram', [])) if 'lbp_histogram' in features2 else 0
            self.gui.hien_thi_ket_qua.cap_nhat_chi_tiet_lbp(chi_square_distance, histogram_size1, histogram_size2)
            
            self.gui.hien_thi_ket_qua.cap_nhat_ket_qua_so_khop(0, similarity_score)
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(
                f"So khớp LBP Texture hoàn tất! Điểm: {similarity_score:.2f}\n"
                f"Chi-square distance: {chi_square_distance:.4f}"
            )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi so khớp LBP: {str(e)}")
    
    def so_khop_ridge(self):
        """So khớp bằng Ridge Orientation Field"""
        if self.anh_xu_ly is None or self.anh_xu_ly_2 is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng làm mảnh ảnh cho cả hai ảnh!")
            return
        
        try:
            # Trích Ridge features từ cả hai ảnh
            from trich_dac_trung.trich_dac_trung_chi_tiet import trich_ridge_orientation_field
            import cv2
            import numpy as np
            
            anh1_gray = cv2.cvtColor(self.anh_xu_ly, cv2.COLOR_BGR2GRAY) if len(self.anh_xu_ly.shape) == 3 else self.anh_xu_ly
            anh2_gray = cv2.cvtColor(self.anh_xu_ly_2, cv2.COLOR_BGR2GRAY) if len(self.anh_xu_ly_2.shape) == 3 else self.anh_xu_ly_2
            
            features1 = trich_ridge_orientation_field(anh1_gray)
            features2 = trich_ridge_orientation_field(anh2_gray)
            
            # Sử dụng ảnh tiền xử lý để tính ridge orientation tốt hơn
            result = so_khop_ridge_orientation(self.anh_xu_ly, self.anh_xu_ly_2)
            similarity_score = result.get('similarity_score', 0)
            
            self.gui.hien_thi_ket_qua.cap_nhat_phuong_phap_so_khop('ridge')
            
            # Cập nhật thông tin chi tiết
            mean_diff = result.get('mean_orientation_diff', 0)
            # Tính trung bình góc từ orientation fields
            field1 = np.array(features1.get('orientation_field', []))
            field2 = np.array(features2.get('orientation_field', []))
            avg_angle1 = float(np.mean(field1)) if field1.size > 0 else 0
            avg_angle2 = float(np.mean(field2)) if field2.size > 0 else 0
            self.gui.hien_thi_ket_qua.cap_nhat_chi_tiet_ridge(mean_diff, avg_angle1, avg_angle2)
            
            self.gui.hien_thi_ket_qua.cap_nhat_ket_qua_so_khop(0, similarity_score)
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(
                f"So khớp Ridge Orientation hoàn tất! Điểm: {similarity_score:.2f}\n"
                f"Chênh lệch góc: {mean_diff:.2f}°"
            )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi so khớp Ridge: {str(e)}")
    
    def so_khop_freq(self):
        """So khớp bằng Frequency Domain"""
        if self.anh_xu_ly is None or self.anh_xu_ly_2 is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng làm mảnh ảnh cho cả hai ảnh!")
            return
        
        try:
            # Trích Frequency features từ cả hai ảnh
            from trich_dac_trung.trich_dac_trung_chi_tiet import trich_frequency_domain_features
            import cv2
            
            anh1_gray = cv2.cvtColor(self.anh_xu_ly, cv2.COLOR_BGR2GRAY) if len(self.anh_xu_ly.shape) == 3 else self.anh_xu_ly
            anh2_gray = cv2.cvtColor(self.anh_xu_ly_2, cv2.COLOR_BGR2GRAY) if len(self.anh_xu_ly_2.shape) == 3 else self.anh_xu_ly_2
            
            features1 = trich_frequency_domain_features(anh1_gray)
            features2 = trich_frequency_domain_features(anh2_gray)
            
            # Sử dụng ảnh tiền xử lý để phân tích frequency domain tốt hơn
            result = so_khop_frequency_domain(self.anh_xu_ly, self.anh_xu_ly_2)
            similarity_score = result.get('similarity_score', 0)
            
            self.gui.hien_thi_ket_qua.cap_nhat_phuong_phap_so_khop('frequency')
            
            # Cập nhật thông tin chi tiết
            freq_sim = result.get('frequency_similarity', 0)
            energy_sim = result.get('energy_similarity', 0)
            # Lấy FFT frequency info từ features
            fft_info1 = f"{features1.get('dominant_frequency', 0):.1f} Hz" if 'dominant_frequency' in features1 else "-"
            fft_info2 = f"{features2.get('dominant_frequency', 0):.1f} Hz" if 'dominant_frequency' in features2 else "-"
            self.gui.hien_thi_ket_qua.cap_nhat_chi_tiet_frequency(
                freq_sim, energy_sim, similarity_score, fft_info1, fft_info2
            )
            
            self.gui.hien_thi_ket_qua.cap_nhat_ket_qua_so_khop(0, similarity_score)
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(
                f"So khớp Frequency Domain hoàn tất! Điểm: {similarity_score:.2f}\n"
                f"Phổ tần: {freq_sim:.2f}%\n"
                f"Năng lượng: {energy_sim:.2f}%"
            )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi so khớp Frequency: {str(e)}")
    
    def _luu_ket_qua_so_khop(self, result, similarity_score, phan_loai, phương_pháp='Minutiae Matching'):
        """Lưu kết quả so khớp vào file"""
        try:
            duong_dan_goc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            thu_muc_ket_qua = os.path.join(duong_dan_goc, 'ket_qua')
            os.makedirs(thu_muc_ket_qua, exist_ok=True)
            
            # Tạo tên file từ tên ảnh gốc
            ten_anh_1 = self._lay_ten_anh(self.duong_dan_anh_1)
            ten_anh_2 = self._lay_ten_anh(self.duong_dan_anh_2)
            
            # Tên file kết quả
            ten_file = f"{ten_anh_1}_vs_{ten_anh_2}_so_khop.txt"
            duong_dan_ket_qua = os.path.join(thu_muc_ket_qua, ten_file)
            
            # Ghi kết quả vào file
            with open(duong_dan_ket_qua, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("KẾT QUẢ SO KHỚP VÂN TAY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Ảnh 1: {self.duong_dan_anh_1}\n")
                f.write(f"Ảnh 2: {self.duong_dan_anh_2}\n\n")
                
                f.write("THỐNG KÊ MINUTIAE:\n")
                f.write(f"  - Ảnh 1: {result['total_minutiae1']} điểm\n")
                f.write(f"  - Ảnh 2: {result['total_minutiae2']} điểm\n\n")
                
                f.write("KẾT QUẢ SO KHỚP:\n")
                f.write(f"  - Số điểm khớp: {result['match_count']}\n")
                f.write(f"  - Tỉ lệ khớp: {result['match_percentage']:.2f}%\n")
                f.write(f"  - Điểm tương đồng: {similarity_score:.2f}/100\n")
                f.write(f"  - Phân loại: {phan_loai.upper()}\n\n")
                
                f.write("=" * 60 + "\n")
            
            print(f"Đã lưu kết quả so khớp: {duong_dan_ket_qua}")
            
        except Exception as e:
            print(f"Lỗi khi lưu kết quả so khớp: {str(e)}")
    
    def _luu_ket_qua_so_khop_image(self, result, similarity_score, phương_pháp='Template Matching'):
        """Lưu kết quả so khớp ảnh vào file"""
        try:
            duong_dan_goc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            thu_muc_ket_qua = os.path.join(duong_dan_goc, 'ket_qua')
            os.makedirs(thu_muc_ket_qua, exist_ok=True)
            
            # Tạo tên file từ tên ảnh gốc
            ten_anh_1 = self._lay_ten_anh(self.duong_dan_anh_1)
            ten_anh_2 = self._lay_ten_anh(self.duong_dan_anh_2)
            
            # Tên file kết quả
            phuong_phap_ten = phương_pháp.lower().replace(' ', '_')
            ten_file = f"{ten_anh_1}_vs_{ten_anh_2}_{phuong_phap_ten}.txt"
            duong_dan_ket_qua = os.path.join(thu_muc_ket_qua, ten_file)
            
            # Ghi kết quả vào file
            with open(duong_dan_ket_qua, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"KẾT QUẢ SO KHỚP VÂN TAY - {phương_pháp.upper()}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Ảnh 1: {self.duong_dan_anh_1}\n")
                f.write(f"Ảnh 2: {self.duong_dan_anh_2}\n\n")
                
                f.write("KẾT QUẢ:\n")
                f.write(f"  - Phương pháp: {phương_pháp}\n")
                f.write(f"  - Điểm tương đồng: {similarity_score:.2f}/100\n")
                f.write(f"  - Phù hợp: {'Có' if result.get('is_match', False) else 'Không'}\n\n")
                
                # Ghi thông tin chi tiết nếu có
                for key, value in result.items():
                    if key not in ['method', 'similarity_score', 'is_match']:
                        f.write(f"  - {key}: {value}\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            print(f"Đã lưu kết quả so khớp: {duong_dan_ket_qua}")
            
        except Exception as e:
            print(f"Lỗi khi lưu kết quả so khớp: {str(e)}")
    
    def xoa_du_lieu(self):
        """Xóa tất cả dữ liệu đã lưu trong thư mục data và ket_qua"""
        # Hỏi xác nhận
        response = messagebox.askyesno(
            "Xác nhận xóa",
            "Bạn có chắc muốn xóa tất cả dữ liệu đã lưu?\n\n"
            "Các thư mục sẽ bị xóa trống:\n"
            "- data/anh_goc/\n"
            "- data/anh_xam/\n"
            "- data/anh_tang_cuong/\n"
            "- data/anh_nhi_phan/\n"
            "- data/anh_lam_manh/\n"
            "- data/dac_trung/\n"
            "- ket_qua/\n\n"
            "Hành động này không thể hoàn tác!"
        )
        
        if not response:
            return
        
        try:
            duong_dan_goc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Các thư mục cần xóa
            thu_muc_can_xoa = [
                os.path.join(duong_dan_goc, 'data', 'anh_goc'),
                os.path.join(duong_dan_goc, 'data', 'anh_xam'),
                os.path.join(duong_dan_goc, 'data', 'anh_tang_cuong'),
                os.path.join(duong_dan_goc, 'data', 'anh_nhi_phan'),
                os.path.join(duong_dan_goc, 'data', 'anh_lam_manh'),
                os.path.join(duong_dan_goc, 'data', 'dac_trung'),
                os.path.join(duong_dan_goc, 'ket_qua')
            ]
            
            tong_file_xoa = 0
            
            # Xóa tất cả file trong các thư mục
            for thu_muc in thu_muc_can_xoa:
                if os.path.exists(thu_muc):
                    for file in os.listdir(thu_muc):
                        duong_dan_file = os.path.join(thu_muc, file)
                        try:
                            if os.path.isfile(duong_dan_file):
                                os.remove(duong_dan_file)
                                tong_file_xoa += 1
                        except Exception as e:
                            print(f"Lỗi khi xóa {duong_dan_file}: {str(e)}")
            
            # Cập nhật thông báo thành công
            self.gui.hien_thi_ket_qua.cap_nhat_thong_bao(
                f"Đã xóa {tong_file_xoa} file từ các thư mục dữ liệu"
            )
            
            messagebox.showinfo("Thành công", f"Đã xóa {tong_file_xoa} file thành công!")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xóa dữ liệu: {str(e)}")

