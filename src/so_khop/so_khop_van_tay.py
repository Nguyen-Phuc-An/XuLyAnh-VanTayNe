"""
Module so khớp ảnh vân tay
"""

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import cv2
from skimage import metrics


def tinh_khoang_cach_minutiae(minutiae1, minutiae2, max_distance=50, angle_tolerance=30):
    """
    Tính khoảng cách giữa hai minutiae points
    Xét cả vị trí và hướng
    
    Args:
        minutiae1 (dict): {'position': (i, j), 'orientation': angle}
        minutiae2 (dict): {'position': (i, j), 'orientation': angle}
        max_distance (float): Khoảng cách tối đa để coi là match
        angle_tolerance (float): Độ chịu nước cơn dành hướng (độ)
        
    Returns:
        float: Điểm tương đồng (0-100), 0 nếu không match
    """
    # Tính khoảng cách Euclidean
    pos1 = np.array(minutiae1['position'])
    pos2 = np.array(minutiae2['position'])
    dist = euclidean(pos1, pos2)
    
    # Kiểm tra khoảng cách
    if dist > max_distance:
        return 0.0
    
    # Tính chênh lệch hướng
    angle1 = minutiae1.get('orientation', 0)
    angle2 = minutiae2.get('orientation', 0)
    angle_diff = abs(angle1 - angle2)
    
    # Điều chỉnh angle_diff về [0, 180]
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    # Kiểm tra hướng
    if angle_diff > angle_tolerance:
        return 0.0
    
    # Tính điểm dựa trên khoảng cách
    distance_score = (max_distance - dist) / max_distance * 100
    
    # Tính điểm dựa trên hướng
    angle_score = (angle_tolerance - angle_diff) / angle_tolerance * 100
    
    # Trung bình hai điểm
    overall_score = (distance_score + angle_score) / 2
    
    return max(0, overall_score)


def so_khop_minutiae(minutiae1, minutiae2, max_distance=50, angle_tolerance=30, min_matches=5):
    """
    So khớp hai tập hợp minutiae
    
    Args:
        minutiae1 (dict): {'endings': [...], 'bifurcations': [...]}
        minutiae2 (dict): {'endings': [...], 'bifurcations': [...]}
        max_distance (float): Khoảng cách tối đa
        angle_tolerance (float): Độ chịu nước cơn về hướng
        min_matches (int): Số match tối thiểu
        
    Returns:
        dict: Kết quả so khớp với chi tiết
    """
    # Kết hợp tất cả minutiae từ cả hai tập hợp
    all_minutiae1 = minutiae1.get('endings', []) + minutiae1.get('bifurcations', [])
    all_minutiae2 = minutiae2.get('endings', []) + minutiae2.get('bifurcations', [])
    
    if not all_minutiae1 or not all_minutiae2:
        return {
            'match_count': 0,
            'total_minutiae1': len(all_minutiae1),
            'total_minutiae2': len(all_minutiae2),
            'similarity_score': 0.0,
            'matched_pairs': []
        }
    
    matched_pairs = []
    matched_indices2 = set()
    
    # Tìm matching cho mỗi minutiae từ tập 1
    for m1 in all_minutiae1:
        best_match = None
        best_score = 0
        best_index = -1
        
        for idx2, m2 in enumerate(all_minutiae2):
            if idx2 in matched_indices2:
                continue
            
            score = tinh_khoang_cach_minutiae(m1, m2, max_distance, angle_tolerance)
            
            if score > best_score:
                best_score = score
                best_match = m2
                best_index = idx2
        
        if best_score > 0:
            matched_pairs.append({
                'minutiae1': m1,
                'minutiae2': best_match,
                'score': best_score
            })
            matched_indices2.add(best_index)
    
    # Tính điểm tương đồng
    if matched_pairs:
        avg_score = np.mean([pair['score'] for pair in matched_pairs])
    else:
        avg_score = 0.0
    
    # Tính tỉ lệ so khớp
    max_minutiae = max(len(all_minutiae1), len(all_minutiae2))
    match_count = len(matched_pairs)
    match_percentage = (match_count / max_minutiae * 100) if max_minutiae > 0 else 0
    
    return {
        'match_count': match_count,
        'total_minutiae1': len(all_minutiae1),
        'total_minutiae2': len(all_minutiae2),
        'similarity_score': avg_score,
        'match_percentage': match_percentage,
        'matched_pairs': matched_pairs,
        'is_match': match_count >= min_matches and avg_score > 30
    }


def so_khop_anh_toan_phan(minutiae1, minutiae2, anh1=None, anh2=None, 
                          max_distance=50, angle_tolerance=30):
    """
    So khớp toàn phần giữa hai ảnh vân tay
    
    Args:
        minutiae1 (dict): Minutiae từ ảnh 1
        minutiae2 (dict): Minutiae từ ảnh 2
        anh1 (np.ndarray): Ảnh làm mảnh 1 (tuỳ chọn)
        anh2 (np.ndarray): Ảnh làm mảnh 2 (tuỳ chọn)
        max_distance (float): Khoảng cách tối đa
        angle_tolerance (float): Độ chịu nước cơn về hướng
        
    Returns:
        dict: Kết quả so khớp chi tiết
    """
    result = so_khop_minutiae(minutiae1, minutiae2, max_distance, angle_tolerance)
    
    # Thêm thông tin ảnh nếu có
    if anh1 is not None and anh2 is not None:
        result['fingerprint1_shape'] = anh1.shape
        result['fingerprint2_shape'] = anh2.shape
    
    return result


def tinh_diem_tuong_dong_tien_tien(minutiae1, minutiae2, 
                                  max_distance=50, angle_tolerance=30):
    """
    Tính điểm tương đồng nâng cao
    Xét thêm cấu trúc và mối quan hệ giữa các minutiae
    
    Args:
        minutiae1 (dict): Minutiae từ ảnh 1
        minutiae2 (dict): Minutiae từ ảnh 2
        max_distance (float): Khoảng cách tối đa
        angle_tolerance (float): Độ chịu nước cơn về hướng
        
    Returns:
        float: Điểm tương đồng cuối cùng (0-100)
    """
    result = so_khop_minutiae(minutiae1, minutiae2, max_distance, angle_tolerance)
    
    match_count = result['match_count']
    total = max(result['total_minutiae1'], result['total_minutiae2'])
    
    if total == 0:
        return 0.0
    
    # Điểm dựa trên số lượng match
    match_score = (match_count / total) * 50
    
    # Điểm dựa trên quality của match
    quality_score = result['similarity_score'] * 0.5
    
    # Điểm cuối cùng
    final_score = min(100, match_score + quality_score)
    
    return final_score


def phan_loai_match(similarity_score, match_percentage):
    """
    Phân loại kết quả so khớp
    
    Args:
        similarity_score (float): Điểm tương đồng (0-100)
        match_percentage (float): Tỉ lệ so khớp (%)
        
    Returns:
        str: Loại match ('match', 'possible_match', 'non_match')
    """
    if similarity_score >= 70 and match_percentage >= 50:
        return 'match'
    elif similarity_score >= 50 and match_percentage >= 30:
        return 'possible_match'
    else:
        return 'non_match'


def so_khop_feature_matching(anh1, anh2, use_sift=True):
    """
    Phương pháp 5: Feature Matching - So khớp dựa trên SIFT/ORB features
    
    Args:
        anh1 (np.ndarray): Ảnh vân tay 1
        anh2 (np.ndarray): Ảnh vân tay 2
        use_sift (bool): Sử dụng SIFT (True) hay ORB (False)
        
    Returns:
        dict: Kết quả so khớp dựa trên features
    """
    if anh1 is None or anh2 is None or anh1.size == 0 or anh2.size == 0:
        return {
            'method': 'feature_matching',
            'similarity_score': 0.0,
            'feature_count1': 0,
            'feature_count2': 0,
            'good_matches': 0,
            'is_match': False
        }
    
    try:
        # Khởi tạo detector
        if use_sift:
            try:
                sift = cv2.SIFT_create()
                detector = sift
            except:
                # SIFT có thể không có sẵn, dùng ORB thay thế
                orb = cv2.ORB_create(nfeatures=500)
                detector = orb
                use_sift = False
        else:
            orb = cv2.ORB_create(nfeatures=500)
            detector = orb
        
        # Phát hiện keypoints và descriptors
        kp1, des1 = detector.detectAndCompute(anh1.astype(np.uint8), None)
        kp2, des2 = detector.detectAndCompute(anh2.astype(np.uint8), None)
        
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return {
                'method': 'feature_matching',
                'similarity_score': 0.0,
                'feature_count1': len(kp1),
                'feature_count2': len(kp2),
                'good_matches': 0,
                'is_match': False
            }
        
        # Sử dụng BFMatcher hoặc FlannBasedMatcher
        if use_sift:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # KNN matching với Lowe's ratio test
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Lọc good matches
        good_matches = []
        if matches is not None:
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:  # Tighter threshold
                        good_matches.append(m)
        
        # Tính similarity score dựa trên số lượng good matches
        # Dùng tổng keypoints từ cả 2 ảnh làm căn cứ, không dùng min
        total_kp = len(kp1) + len(kp2)
        if total_kp > 0:
            # Tỷ lệ match: số good matches chia cho tổng keypoints
            # Nếu 2 ảnh y chang, số good matches ≈ min(len(kp1), len(kp2))
            # Nên similarity ≈ min(len(kp1), len(kp2)) / (len(kp1) + len(kp2))
            # Ví dụ: 2000 + 2000 kp, 2000 good match → 2000/4000 = 50%
            # Nếu 2 ảnh khác, 2334 + 40 kp, 41 good match → 41/2374 = 1.7%
            match_ratio = len(good_matches) / total_kp
            similarity_score = match_ratio * 100
        else:
            similarity_score = 0.0
        
        return {
            'method': 'feature_matching',
            'similarity_score': max(0, min(100, similarity_score)),
            'feature_count1': len(kp1),
            'feature_count2': len(kp2),
            'good_matches': len(good_matches),
            'match_ratio': match_ratio if total_kp > 0 else 0,
            'is_match': len(good_matches) >= 5 and similarity_score >= 30
        }
    except Exception as e:
        return {
            'method': 'feature_matching',
            'similarity_score': 0.0,
            'error': str(e),
            'is_match': False
        }


def so_khop_thong_ke_toan_bo(minutiae1, minutiae2, anh1=None, anh2=None):
    """
    Phương pháp 6: Comprehensive Matching - So khớp toàn diện kết hợp nhiều phương pháp
    
    Args:
        minutiae1 (dict): Minutiae từ ảnh 1
        minutiae2 (dict): Minutiae từ ảnh 2
        anh1 (np.ndarray): Ảnh làm mảnh 1 (tuỳ chọn)
        anh2 (np.ndarray): Ảnh làm mảnh 2 (tuỳ chọn)
        
    Returns:
        dict: Kết quả so khớp từ tất cả phương pháp
    """
    results = {
        'minutiae_matching': so_khop_minutiae(minutiae1, minutiae2)
    }
    
    if anh1 is not None and anh2 is not None:
        results['feature_matching'] = so_khop_feature_matching(anh1, anh2)
        results['lbp_texture'] = so_khop_lbp_texture(anh1, anh2)
        results['ridge_orientation'] = so_khop_ridge_orientation(anh1, anh2)
        results['frequency_domain'] = so_khop_frequency_domain(anh1, anh2)
        
        # Tính điểm trung bình từ tất cả phương pháp
        scores = []
        for method, result in results.items():
            if 'similarity_score' in result:
                scores.append(result['similarity_score'])
        
        if scores:
            results['overall_score'] = np.mean(scores)
            results['max_score'] = np.max(scores)
            results['min_score'] = np.min(scores)
        else:
            results['overall_score'] = 0.0
            results['max_score'] = 0.0
            results['min_score'] = 0.0
    
    return results


# ============================================================================
# CÁC PHƯƠNG PHÁP SO KHỚP BỔ SUNG CHO CÁC ĐẶC TRƯNG MỚI
# ============================================================================


def so_khop_lbp_texture(anh1, anh2):
    """
    So khớp sử dụng Local Binary Pattern (LBP)
    So sánh histogram LBP của hai ảnh
    
    Args:
        anh1 (np.ndarray): Ảnh 1
        anh2 (np.ndarray): Ảnh 2
        
    Returns:
        dict: Kết quả so khớp
    """
    from trich_dac_trung.trich_dac_trung_chi_tiet import trich_lbp_features
    
    # Convert to grayscale if needed
    if len(anh1.shape) == 3:
        anh1_gray = cv2.cvtColor(anh1, cv2.COLOR_BGR2GRAY)
    else:
        anh1_gray = anh1
    
    if len(anh2.shape) == 3:
        anh2_gray = cv2.cvtColor(anh2, cv2.COLOR_BGR2GRAY)
    else:
        anh2_gray = anh2
    
    features1 = trich_lbp_features(anh1_gray)
    features2 = trich_lbp_features(anh2_gray)
    
    hist1 = np.array(features1['lbp_histogram'], dtype=np.float32)
    hist2 = np.array(features2['lbp_histogram'], dtype=np.float32)
    
    # Chuẩn hóa histograms
    hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
    hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
    
    # Tính Chi-square distance
    chi_square = np.sum((hist1_norm - hist2_norm) ** 2 / (hist1_norm + hist2_norm + 1e-10))
    
    # Chuyển đổi thành similarity score (0-100)
    # Chi-square nhỏ = hình giống nhau
    # Chi-square thường trong khoảng 0-2 đối với 2 ảnh khác nhau
    # Sử dụng exponential decay thay vì linear
    # similarity = 100 * exp(-chi_square / 2)
    # Nếu chi_square = 0 → similarity = 100
    # Nếu chi_square = 2 → similarity ≈ 37
    similarity = 100 * np.exp(-chi_square / 2)
    
    return {
        'method': 'LBP Texture',
        'similarity_score': max(0, min(100, similarity)),
        'texture_uniformity_1': features1['texture_uniformity'],
        'texture_uniformity_2': features2['texture_uniformity'],
        'chi_square_distance': chi_square
    }


def so_khop_ridge_orientation(anh1, anh2):
    """
    So khớp sử dụng Ridge Orientation Field
    So sánh hướng của các sọc vân tay
    
    Args:
        anh1 (np.ndarray): Ảnh 1
        anh2 (np.ndarray): Ảnh 2
        
    Returns:
        dict: Kết quả so khớp
    """
    from trich_dac_trung.trich_dac_trung_chi_tiet import trich_ridge_orientation_field
    
    # Convert to grayscale if needed
    if len(anh1.shape) == 3:
        anh1_gray = cv2.cvtColor(anh1, cv2.COLOR_BGR2GRAY)
    else:
        anh1_gray = anh1
    
    if len(anh2.shape) == 3:
        anh2_gray = cv2.cvtColor(anh2, cv2.COLOR_BGR2GRAY)
    else:
        anh2_gray = anh2
    
    features1 = trich_ridge_orientation_field(anh1_gray)
    features2 = trich_ridge_orientation_field(anh2_gray)
    
    field1 = np.array(features1['orientation_field'])
    field2 = np.array(features2['orientation_field'])
    
    # Đảm bảo cùng kích thước
    min_h = min(field1.shape[0], field2.shape[0])
    min_w = min(field1.shape[1], field2.shape[1])
    
    field1 = field1[:min_h, :min_w]
    field2 = field2[:min_h, :min_w]
    
    # Tính sự khác biệt góc (0-90 độ)
    angle_diff = np.abs(field1 - field2)
    angle_diff = np.minimum(angle_diff, 180 - angle_diff)
    
    # Tính mean difference
    mean_diff = np.mean(angle_diff)
    
    # Chuyển đổi thành similarity (0-100)
    # Nếu mean_diff nhỏ (gần 0) thì similarity cao
    # Max mean_diff là 90 độ
    # Sử dụng exponential decay thay vì linear
    # similarity = 100 * exp(-mean_diff / 45)
    # Nếu mean_diff = 0 → similarity = 100
    # Nếu mean_diff = 45 → similarity ≈ 37
    # Nếu mean_diff = 90 → similarity ≈ 14
    similarity = 100 * np.exp(-mean_diff / 45)
    
    return {
        'method': 'Ridge Orientation',
        'similarity_score': similarity,
        'mean_orientation_diff': float(mean_diff)
    }


def so_khop_frequency_domain(anh1, anh2):
    """
    So khớp sử dụng Frequency Domain Analysis
    So sánh các đặc trưng tần số
    
    Args:
        anh1 (np.ndarray): Ảnh 1
        anh2 (np.ndarray): Ảnh 2
        
    Returns:
        dict: Kết quả so khớp
    """
    from trich_dac_trung.trich_dac_trung_chi_tiet import trich_frequency_domain_features
    
    # Convert to grayscale if needed
    if len(anh1.shape) == 3:
        anh1_gray = cv2.cvtColor(anh1, cv2.COLOR_BGR2GRAY)
    else:
        anh1_gray = anh1
    
    if len(anh2.shape) == 3:
        anh2_gray = cv2.cvtColor(anh2, cv2.COLOR_BGR2GRAY)
    else:
        anh2_gray = anh2
    
    features1 = trich_frequency_domain_features(anh1_gray)
    features2 = trich_frequency_domain_features(anh2_gray)
    
    # So sánh các đặc trưng tần số
    freq_diff = abs(features1['dominant_frequency'] - features2['dominant_frequency'])
    energy_diff = abs(features1['energy_concentration'] - features2['energy_concentration'])
    ridge_freq_diff = abs(features1['ridge_frequency'] - features2['ridge_frequency'])
    
    # Tính similarity dựa trên các đặc trưng với exponential decay
    # Frequency diff: max là ~50Hz
    freq_similarity = 100 * np.exp(-freq_diff / 25)
    
    # Energy diff: max là ~1.0
    energy_similarity = 100 * np.exp(-abs(energy_diff) * 5)
    
    # Ridge frequency diff: max là ~0.5
    ridge_similarity = 100 * np.exp(-ridge_freq_diff * 20)
    
    # Lấy trung bình, nhưng yêu cầu tất cả 3 đặc trưng phải tương đồng
    # Nếu 1 đặc trưng quá khác nhau → similarity sẽ bị hạ
    overall_similarity = (freq_similarity + energy_similarity + ridge_similarity) / 3
    overall_similarity = min(overall_similarity, 100)
    
    return {
        'method': 'Frequency Domain',
        'similarity_score': overall_similarity,
        'frequency_similarity': freq_similarity,
        'energy_similarity': energy_similarity,
        'ridge_similarity': ridge_similarity,
        'dominant_freq_1': features1['dominant_frequency'],
        'dominant_freq_2': features2['dominant_frequency']
    }