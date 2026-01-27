"""
Module xử lý sự kiện liên quan đến database
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database_manager import DatabaseManager
from nhan_dang.fingerprint_recognition import FingerprintRecognition


class DatabaseEventHandler:
    """Xử lý sự kiện liên quan đến database"""
    
    def __init__(self, gui):
        """
        Khởi tạo handler database
        
        Args:
            gui: Reference tới GiaoDienChinh
        """
        self.gui = gui
        self.db = None
        self.recognition = None
        self.is_connected = False
        self.xu_ly_su_kien = None  # Reference tới XuLySuKien
    
    def set_xu_ly_su_kien(self, xu_ly_su_kien):
        """Thiết lập reference tới XuLySuKien"""
        self.xu_ly_su_kien = xu_ly_su_kien
    
    def ket_noi_database(self, host='localhost', user='root', password='123456', database='xla_vantay'):
        """Kết nối tới database"""
        try:
            self.db = DatabaseManager(host, user, password, database)
            if self.db.connect():
                self.recognition = FingerprintRecognition(self.db)
                self.is_connected = True
                messagebox.showinfo("Thành công", "Kết nối database thành công!")
                return True
            else:
                messagebox.showerror("Lỗi", "Không thể kết nối database.\nVui lòng kiểm tra cấu hình MySQL.")
                return False
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi kết nối database: {str(e)}")
            return False
    
    def kiểm_tra_kết_nối(self):
        """Kiểm tra kết nối database"""
        if not self.is_connected or not self.db:
            messagebox.showwarning("Cảnh báo", "Chưa kết nối database!\nVui lòng kết nối trước.")
            return False
        return True
    
    def them_nguoi_dung_moi(self, username, full_name, email, phone, address,
                           date_of_birth, gender, identification_number,
                           position, department, notes):
        """Thêm người dùng mới vào database"""
        if not self.kiểm_tra_kết_nối():
            return None
        
        try:
            # Kiểm tra username đã tồn tại chưa
            if self.db.get_user_by_username(username):
                messagebox.showwarning("Cảnh báo", "Username đã tồn tại!")
                return None
            
            user_id = self.db.add_user(
                username, full_name, email, phone, address,
                date_of_birth, gender, identification_number,
                position, department, notes
            )
            
            if user_id:
                messagebox.showinfo("Thành công", f"Thêm người dùng thành công! (ID: {user_id})")
                return user_id
            else:
                messagebox.showerror("Lỗi", "Lỗi khi thêm người dùng!")
                return None
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi: {str(e)}")
            return None
    
    def luu_van_tay(self, user_id, finger_name, hand, 
                    anh_xu_ly, minutiae_data=None, quality_score=None):
        """Lưu vân tay vào database (lưu ảnh nhị phân để trích features khi cần)"""
        if not self.kiểm_tra_kết_nối():
            return None
        
        try:
            # Encode ảnh nhị phân từ numpy array
            binary_image_data = None
            if anh_xu_ly is not None:
                _, binary_image_data = cv2.imencode('.png', anh_xu_ly)
                binary_image_data = binary_image_data.tobytes()
            
            fingerprint_id = self.db.add_fingerprint(
                user_id, finger_name, hand,
                minutiae_data=minutiae_data,
                binary_image_data=binary_image_data,
                quality_score=quality_score
            )
            
            if fingerprint_id:
                messagebox.showinfo("Thành công", f"Lưu vân tay thành công! (ID: {fingerprint_id})")
                return fingerprint_id
            else:
                messagebox.showerror("Lỗi", "Lỗi khi lưu vân tay!")
                return None
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi: {str(e)}")
            return None
    
    def nhan_dang_van_tay(self, anh_manh, minutiae_data, matching_method='comprehensive', anh_xu_ly=None):
        """Nhận dạng người dùng từ vân tay"""
        if not self.kiểm_tra_kết_nối():
            return []
        
        try:
            # Thiết lập ngưỡng từ cài đặt
            threshold = float(self.db.get_setting('matching_threshold') or 70.0)
            self.recognition.set_threshold(threshold)
            
            # Nhận dạng (truyền anh_xu_ly để trích features từ query)
            results = self.recognition.identify_user_from_image(
                anh_manh, minutiae_data, matching_method, anh_xu_ly=anh_xu_ly
            )
            
            return results
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi nhận dạng: {str(e)}")
            return []
    
    def lay_thong_tin_nguoi_dung(self, user_id):
        """Lấy thông tin chi tiết người dùng"""
        if not self.kiểm_tra_kết_nối():
            return None
        
        try:
            return self.recognition.get_user_info(user_id)
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            return None
    
    def lay_danh_sach_nguoi_dung(self):
        """Lấy danh sách tất cả người dùng"""
        if not self.kiểm_tra_kết_nối():
            return []
        
        try:
            return self.db.get_all_users('active')
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            return []
    
    def tim_kiem_nguoi_dung(self, keyword):
        """Tìm kiếm người dùng"""
        if not self.kiểm_tra_kết_nối():
            return []
        
        try:
            return self.db.search_users(keyword)
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            return []
    
    def cap_nhat_van_tay_match(self, fingerprint_id, query_image_path,
                              matching_method, similarity_score, is_match):
        """Lưu bản ghi so khớp vào database"""
        if not self.kiểm_tra_kết_nối():
            return False
        
        try:
            # Lấy thông tin fingerprint
            fp = self.db.get_fingerprint_by_id(fingerprint_id)
            if not fp:
                return False
            
            user_id = fp['user_id']
            
            # Lưu bản ghi
            self.recognition.save_match_record(
                user_id, fingerprint_id, query_image_path,
                matching_method, similarity_score, is_match
            )
            return True
        
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            return False
    
    def lay_thong_ke(self):
        """Lấy thống kê hệ thống"""
        if not self.kiểm_tra_kết_nối():
            return {}
        
        try:
            return self.db.get_statistics()
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            return {}
    
    def lay_lich_su_so_khop(self, user_id=None, limit=100):
        """Lấy lịch sử so khớp"""
        if not self.kiểm_tra_kết_nối():
            return []
        
        try:
            return self.db.get_matching_history(user_id, limit)
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            return []
