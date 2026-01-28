"""
Module nhận dạng người dùng từ vân tay
"""

import numpy as np
import os
from database.database_manager import DatabaseManager
from so_khop.so_khop_van_tay import (
    so_khop_minutiae, so_khop_feature_matching,
    so_khop_lbp_texture, so_khop_ridge_orientation,
    so_khop_frequency_domain
)


class FingerprintRecognition:
    """Nhận dạng người dùng từ vân tay"""
    
    def __init__(self, db_manager):
        """
        Khởi tạo hệ thống nhận dạng
        
        Args:
            db_manager: Instance của DatabaseManager
        """
        self.db = db_manager
        self.threshold = 70.0  # Ngưỡng mặc định
    
    def set_threshold(self, threshold):
        """Thiết lập ngưỡng so khớp"""
        self.threshold = threshold
    
    def _get_threshold_for_method(self, method):
        """
        Lấy ngưỡng phù hợp cho từng phương pháp so khớp
        
        Args:
            method (str): Phương pháp so khớp
            
        Returns:
            float: Ngưỡng phù hợp
        """
        # Ngưỡng khác nhau cho các phương pháp khác nhau
        # Giảm threshold để các phương pháp có thể tìm thấy matches
        method_thresholds = {
            'minutiae': 70.0,
            'feature': 50.0,
            'lbp': 50.0,
            'ridge': 50.0,
            'frequency': 50.0
        }
        
        return method_thresholds.get(method, self.threshold)
    
    def identify_user_from_minutiae(self, minutiae, max_results=5):
        """
        Nhận dạng người dùng từ minutiae
        
        Args:
            minutiae: Dữ liệu minutiae từ ảnh vân tay
            max_results: Số lượng kết quả tối đa
            
        Returns:
            Danh sách những người dùng khớp nhất
        """
        results = []
        
        # Lấy tất cả vân tay từ database
        db_fingerprints = self.db.get_fingerprints_for_matching()
        
        if not db_fingerprints:
            return results
        
        # So khớp với từng vân tay trong database
        for db_fp in db_fingerprints:
            fingerprint_id = db_fp['fingerprint_id']
            user_id = db_fp['user_id']
            username = db_fp['username']
            full_name = db_fp['full_name']
            finger_name = db_fp['finger_name']
            
            # Lấy dữ liệu minutiae từ database
            try:
                import json
                db_minutiae = json.loads(db_fp['minutiae_data']) if db_fp['minutiae_data'] else None
                
                if not db_minutiae:
                    continue
                
                # So khớp minutiae
                match_result = so_khop_minutiae(minutiae, db_minutiae)
                similarity_score = match_result.get('similarity_score', 0)
                
                if similarity_score > self.threshold:
                    results.append({
                        'user_id': user_id,
                        'username': username,
                        'full_name': full_name,
                        'fingerprint_id': fingerprint_id,
                        'finger_name': finger_name,
                        'similarity_score': similarity_score,
                        'method': 'minutiae_matching'
                    })
            
            except Exception as e:
                print(f"Lỗi so khớp minutiae với fingerprint {fingerprint_id}: {e}")
                continue
        
        # Sắp xếp theo điểm tương đồng giảm dần
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:max_results]
    
    def identify_user_from_image(self, image, minutiae, matching_method='minutiae', max_results=5, anh_xu_ly=None):
        """
        Nhận dạng người dùng từ ảnh vân tay
        So sánh các features đã trích và lưu trong database
        
        Args:
            image: Ảnh vân tay làm mảnh
            minutiae: Dữ liệu minutiae
            matching_method: Phương pháp so khớp ('minutiae', 'feature', 'lbp', 'ridge', 'frequency')
            max_results: Số lượng kết quả tối đa
            anh_xu_ly: Ảnh nhị phân (cho trích features từ query)
            
        Returns:
            Danh sách những người dùng khớp nhất
        """
        results = []
        best_matches = []
        
        # Lấy ngưỡng phù hợp cho phương pháp
        method_threshold = self._get_threshold_for_method(matching_method)
        
        # Lấy tất cả vân tay từ database
        db_fingerprints = self.db.get_fingerprints_for_matching()
        
        if not db_fingerprints:
            return results
        
        # So khớp với từng vân tay trong database
        for db_fp in db_fingerprints:
            fingerprint_id = db_fp['fingerprint_id']
            user_id = db_fp['user_id']
            username = db_fp['username']
            full_name = db_fp['full_name']
            finger_name = db_fp['finger_name']
            
            try:
                import json
                import cv2
                
                similarity_score = 0
                
                # Decode ảnh nhị phân từ database
                binary_image_data = db_fp.get('binary_image_data')
                db_binary_image = None
                
                if isinstance(binary_image_data, bytes):
                    nparr = np.frombuffer(binary_image_data, np.uint8)
                    db_binary_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                # Áp dụng phương pháp so khớp
                if matching_method == 'minutiae':
                    db_minutiae = json.loads(db_fp['minutiae_data']) if db_fp['minutiae_data'] else None
                    if db_minutiae and minutiae:
                        match_result = so_khop_minutiae(minutiae, db_minutiae)
                        similarity_score = match_result.get('similarity_score', 0)
                
                elif matching_method == 'feature':
                    # Dùng hàm so khớp có sẵn
                    if anh_xu_ly is not None and db_binary_image is not None:
                        match_result = so_khop_feature_matching(anh_xu_ly, db_binary_image)
                        similarity_score = match_result.get('similarity_score', 0)
                
                elif matching_method == 'lbp':
                    # Dùng hàm so khớp có sẵn
                    if anh_xu_ly is not None and db_binary_image is not None:
                        match_result = so_khop_lbp_texture(anh_xu_ly, db_binary_image)
                        similarity_score = match_result.get('similarity_score', 0)
                
                elif matching_method == 'ridge':
                    # Dùng hàm so khớp có sẵn
                    if anh_xu_ly is not None and db_binary_image is not None:
                        match_result = so_khop_ridge_orientation(anh_xu_ly, db_binary_image)
                        similarity_score = match_result.get('similarity_score', 0)
                
                elif matching_method == 'frequency':
                    # Dùng hàm so khớp có sẵn
                    if anh_xu_ly is not None and db_binary_image is not None:
                        match_result = so_khop_frequency_domain(anh_xu_ly, db_binary_image)
                        similarity_score = match_result.get('similarity_score', 0)
                
                else:
                    # Mặc định là minutiae
                    db_minutiae = json.loads(db_fp['minutiae_data']) if db_fp['minutiae_data'] else None
                    if db_minutiae and minutiae:
                        match_result = so_khop_minutiae(minutiae, db_minutiae)
                        similarity_score = match_result.get('similarity_score', 0)
                
                match_result = {
                    'user_id': user_id,
                    'username': username,
                    'full_name': full_name,
                    'fingerprint_id': fingerprint_id,
                    'finger_name': finger_name,
                    'similarity_score': similarity_score,
                    'method': matching_method
                }
                
                # Lưu vào danh sách kết quả tốt nhất
                best_matches.append(match_result)
                
                # Nếu vượt qua threshold, thêm vào kết quả chính thức
                if similarity_score > method_threshold:
                    results.append(match_result)
            
            except Exception as e:
                print(f"Lỗi so khớp với fingerprint {fingerprint_id}: {e}")
                continue
        
        # Sắp xếp danh sách kết quả tốt nhất theo điểm giảm dần
        best_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Nếu không tìm thấy kết quả vượt qua threshold, trả về kết quả tốt nhất
        if not results and best_matches:
            results = best_matches[:max_results]
        elif results:
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            results = results[:max_results]
        
        return results
    
    def get_user_info(self, user_id):
        """Lấy thông tin chi tiết của người dùng"""
        user = self.db.get_user_by_id(user_id)
        if user:
            user['fingerprints'] = self.db.get_fingerprints_by_user(user_id)
        return user
    
    def save_match_record(self, user_id, fingerprint_id, query_image_path,
                         matching_method, similarity_score, is_match):
        """Lưu bản ghi so khớp vào database"""
        return self.db.add_matching_record(
            user_id, fingerprint_id, query_image_path,
            matching_method, similarity_score, is_match
        )
