-- Tạo database cho hệ thống nhận dạng vân tay
CREATE DATABASE IF NOT EXISTS xla_vantay;
USE xla_vantay;

-- Table người dùng
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    full_name VARCHAR(150) NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(20),
    address VARCHAR(255),
    date_of_birth DATE,
    gender ENUM('Nam', 'Nữ', 'Khác'),
    identification_number VARCHAR(20),
    position VARCHAR(100),
    department VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status ENUM('active', 'inactive') DEFAULT 'active',
    notes TEXT,
    INDEX idx_username (username),
    INDEX idx_identification (identification_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table vân tay
CREATE TABLE IF NOT EXISTS fingerprints (
    fingerprint_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    finger_name VARCHAR(50) NOT NULL COMMENT 'Tên ngón tay: Thumb, Index, Middle, Ring, Pinky, etc',
    hand ENUM('Left', 'Right') NOT NULL COMMENT 'Tay trái hay tay phải',
    minutiae_data JSON COMMENT 'Minutiae features (endings, bifurcations)',
    binary_image_data LONGBLOB COMMENT 'Binary image for feature extraction',
    quality_score FLOAT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('approved', 'pending', 'rejected') DEFAULT 'approved',
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_finger_hand (finger_name, hand),
    UNIQUE KEY unique_user_finger (user_id, finger_name, hand)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table lịch sử so khớp
CREATE TABLE IF NOT EXISTS matching_history (
    match_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    fingerprint_id INT,
    query_image_path VARCHAR(255),
    matching_method VARCHAR(100),
    similarity_score FLOAT,
    is_match BOOLEAN,
    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (fingerprint_id) REFERENCES fingerprints(fingerprint_id) ON DELETE SET NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_matched_at (matched_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table cấu hình hệ thống
CREATE TABLE IF NOT EXISTS system_settings (
    setting_id INT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) NOT NULL UNIQUE,
    setting_value VARCHAR(255),
    setting_type VARCHAR(50),
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert một số cài đặt mặc định
INSERT INTO system_settings (setting_key, setting_value, setting_type, description) VALUES
('matching_threshold', '70', 'float', 'Ngưỡng điểm tương đồng để coi là khớp (0-100)'),
('template_matching_threshold', '50', 'float', 'Ngưỡng cho Template Matching'),
('ssim_threshold', '50', 'float', 'Ngưỡng cho SSIM'),
('contour_threshold', '50', 'float', 'Ngưỡng cho Contour Matching'),
('histogram_threshold', '50', 'float', 'Ngưỡng cho Histogram Matching'),
('feature_threshold', '50', 'float', 'Ngưỡng cho Feature Matching'),
('max_results', '5', 'int', 'Số lượng kết quả tối đa trả về'),
('default_matching_method', 'comprehensive', 'string', 'Phương pháp so khớp mặc định');
