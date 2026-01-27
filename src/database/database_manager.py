"""
Module xử lý database MySQL cho hệ thống nhận dạng vân tay
"""

import mysql.connector
from mysql.connector import Error
import json
import os
import cv2
import numpy as np
from datetime import datetime


class DatabaseManager:
    """Quản lý kết nối và các thao tác với database"""
    
    def __init__(self, host='localhost', user='root', password='123456', database='xla_vantay'):
        """
        Khởi tạo kết nối database
        
        Args:
            host: Địa chỉ host MySQL
            user: Tên user MySQL
            password: Password MySQL
            database: Tên database
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Kết nối tới database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor(dictionary=True)
            print("✓ Kết nối database thành công")
            return True
        except Error as e:
            print(f"✗ Lỗi kết nối database: {e}")
            return False
    
    def disconnect(self):
        """Ngắt kết nối database"""
        if self.connection and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("✓ Đã ngắt kết nối database")
    
    def execute_query(self, query, params=None):
        """
        Thực thi query
        
        Args:
            query: Câu lệnh SQL
            params: Tham số cho câu lệnh
            
        Returns:
            Kết quả query
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            self.connection.commit()
            return True
        except Error as e:
            print(f"Lỗi thực thi query: {e}")
            self.connection.rollback()
            return False
    
    def fetch_query(self, query, params=None):
        """
        Lấy kết quả từ query
        
        Args:
            query: Câu lệnh SQL
            params: Tham số cho câu lệnh
            
        Returns:
            Kết quả query
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except Error as e:
            print(f"Lỗi lấy dữ liệu: {e}")
            return None
    
    # ==================== USER OPERATIONS ====================
    
    def add_user(self, username, full_name, email=None, phone=None, address=None,
                 date_of_birth=None, gender=None, identification_number=None,
                 position=None, department=None, notes=None):
        """Thêm người dùng mới"""
        query = """
        INSERT INTO users (username, full_name, email, phone, address, 
                          date_of_birth, gender, identification_number, 
                          position, department, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (username, full_name, email, phone, address,
                 date_of_birth, gender, identification_number,
                 position, department, notes)
        
        if self.execute_query(query, params):
            return self.cursor.lastrowid
        return None
    
    def get_user_by_id(self, user_id):
        """Lấy thông tin người dùng theo ID"""
        query = "SELECT * FROM users WHERE user_id = %s"
        result = self.fetch_query(query, (user_id,))
        return result[0] if result else None
    
    def get_user_by_username(self, username):
        """Lấy thông tin người dùng theo username"""
        query = "SELECT * FROM users WHERE username = %s"
        result = self.fetch_query(query, (username,))
        return result[0] if result else None
    
    def get_all_users(self, status='active'):
        """Lấy danh sách tất cả người dùng"""
        query = "SELECT * FROM users WHERE status = %s ORDER BY created_at DESC"
        return self.fetch_query(query, (status,))
    
    def update_user(self, user_id, **kwargs):
        """Cập nhật thông tin người dùng"""
        allowed_fields = ['username', 'full_name', 'email', 'phone', 'address',
                         'date_of_birth', 'gender', 'identification_number',
                         'position', 'department', 'status', 'notes']
        
        fields = [f"{key} = %s" for key in kwargs.keys() if key in allowed_fields]
        if not fields:
            return False
        
        values = [kwargs[key] for key in kwargs.keys() if key in allowed_fields]
        values.append(user_id)
        
        query = f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s"
        return self.execute_query(query, values)
    
    def delete_user(self, user_id):
        """Xóa người dùng"""
        query = "DELETE FROM users WHERE user_id = %s"
        return self.execute_query(query, (user_id,))
    
    # ==================== FINGERPRINT OPERATIONS ====================
    
    def add_fingerprint(self, user_id, finger_name, hand, 
                       minutiae_data=None, binary_image_data=None,
                       quality_score=None):
        """Thêm vân tay mới"""
        query = """
        INSERT INTO fingerprints (user_id, finger_name, hand, 
                                 minutiae_data, binary_image_data, quality_score)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        # Chuẩn bị dữ liệu JSON
        minutiae_json = json.dumps(minutiae_data) if minutiae_data else None
        
        params = (user_id, finger_name, hand, 
                 minutiae_json, binary_image_data, quality_score)
        
        if self.execute_query(query, params):
            return self.cursor.lastrowid
        return None
    
    def get_fingerprints_by_user(self, user_id):
        """Lấy tất cả vân tay của một người dùng"""
        query = "SELECT * FROM fingerprints WHERE user_id = %s ORDER BY captured_at DESC"
        return self.fetch_query(query, (user_id,))
    
    def get_fingerprint_by_id(self, fingerprint_id):
        """Lấy thông tin vân tay theo ID"""
        query = "SELECT * FROM fingerprints WHERE fingerprint_id = %s"
        result = self.fetch_query(query, (fingerprint_id,))
        return result[0] if result else None
    
    def get_all_fingerprints(self, status='approved'):
        """Lấy tất cả vân tay"""
        query = "SELECT * FROM fingerprints WHERE status = %s ORDER BY captured_at DESC"
        return self.fetch_query(query, (status,))
    
    def update_fingerprint(self, fingerprint_id, **kwargs):
        """Cập nhật thông tin vân tay"""
        allowed_fields = ['finger_name', 'hand', 'image_path', 'image_data',
                         'minutiae_data', 'feature_data', 'lbp_data', 'ridge_data',
                         'frequency_data', 'quality_score', 'status', 'notes']
        
        fields = [f"{key} = %s" for key in kwargs.keys() if key in allowed_fields]
        if not fields:
            return False
        
        values = [kwargs[key] for key in kwargs.keys() if key in allowed_fields]
        values.append(fingerprint_id)
        
        query = f"UPDATE fingerprints SET {', '.join(fields)} WHERE fingerprint_id = %s"
        return self.execute_query(query, values)
    
    def delete_fingerprint(self, fingerprint_id):
        """Xóa vân tay"""
        query = "DELETE FROM fingerprints WHERE fingerprint_id = %s"
        return self.execute_query(query, (fingerprint_id,))
    
    # ==================== MATCHING OPERATIONS ====================
    
    def add_matching_record(self, user_id, fingerprint_id, query_image_path,
                           matching_method, similarity_score, is_match, notes=None):
        """Thêm bản ghi lịch sử so khớp"""
        query = """
        INSERT INTO matching_history (user_id, fingerprint_id, query_image_path,
                                      matching_method, similarity_score, is_match, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (user_id, fingerprint_id, query_image_path,
                 matching_method, similarity_score, is_match, notes)
        
        if self.execute_query(query, params):
            return self.cursor.lastrowid
        return None
    
    def get_matching_history(self, user_id=None, limit=100):
        """Lấy lịch sử so khớp"""
        if user_id:
            query = """
            SELECT mh.*, u.username, u.full_name, f.finger_name
            FROM matching_history mh
            LEFT JOIN users u ON mh.user_id = u.user_id
            LEFT JOIN fingerprints f ON mh.fingerprint_id = f.fingerprint_id
            WHERE mh.user_id = %s
            ORDER BY mh.matched_at DESC
            LIMIT %s
            """
            return self.fetch_query(query, (user_id, limit))
        else:
            query = """
            SELECT mh.*, u.username, u.full_name, f.finger_name
            FROM matching_history mh
            LEFT JOIN users u ON mh.user_id = u.user_id
            LEFT JOIN fingerprints f ON mh.fingerprint_id = f.fingerprint_id
            ORDER BY mh.matched_at DESC
            LIMIT %s
            """
            return self.fetch_query(query, (limit,))
    
    # ==================== SETTINGS OPERATIONS ====================
    
    def get_setting(self, setting_key):
        """Lấy cài đặt"""
        query = "SELECT setting_value FROM system_settings WHERE setting_key = %s"
        result = self.fetch_query(query, (setting_key,))
        return result[0]['setting_value'] if result else None
    
    def update_setting(self, setting_key, setting_value):
        """Cập nhật cài đặt"""
        query = """
        UPDATE system_settings SET setting_value = %s WHERE setting_key = %s
        """
        return self.execute_query(query, (setting_value, setting_key))
    
    def get_all_settings(self):
        """Lấy tất cả cài đặt"""
        query = "SELECT * FROM system_settings"
        return self.fetch_query(query)
    
    # ==================== SEARCH OPERATIONS ====================
    
    def search_users(self, keyword):
        """Tìm kiếm người dùng"""
        query = """
        SELECT * FROM users
        WHERE username LIKE %s OR full_name LIKE %s OR email LIKE %s
        ORDER BY created_at DESC
        """
        keyword_pattern = f"%{keyword}%"
        return self.fetch_query(query, (keyword_pattern, keyword_pattern, keyword_pattern))
    
    def get_fingerprints_for_matching(self):
        """Lấy danh sách vân tay để so khớp"""
        query = """
        SELECT f.*, u.username, u.full_name
        FROM fingerprints f
        JOIN users u ON f.user_id = u.user_id
        WHERE f.status = 'approved' AND u.status = 'active'
        ORDER BY u.username, f.finger_name
        """
        return self.fetch_query(query)
    
    def get_statistics(self):
        """Lấy thống kê hệ thống"""
        stats = {}
        
        # Tổng số người dùng
        result = self.fetch_query("SELECT COUNT(*) as count FROM users WHERE status = 'active'")
        stats['total_users'] = result[0]['count'] if result else 0
        
        # Tổng số vân tay
        result = self.fetch_query("SELECT COUNT(*) as count FROM fingerprints WHERE status = 'approved'")
        stats['total_fingerprints'] = result[0]['count'] if result else 0
        
        # Tổng số lần so khớp
        result = self.fetch_query("SELECT COUNT(*) as count FROM matching_history")
        stats['total_matches'] = result[0]['count'] if result else 0
        
        # Tỉ lệ match thành công
        result = self.fetch_query("SELECT COUNT(*) as count FROM matching_history WHERE is_match = TRUE")
        successful_matches = result[0]['count'] if result else 0
        stats['successful_matches'] = successful_matches
        stats['match_success_rate'] = (successful_matches / stats['total_matches'] * 100) if stats['total_matches'] > 0 else 0
        
        return stats
