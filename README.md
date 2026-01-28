# Hệ Thống Nhận Dạng Vân Tay

Một hệ thống nhận dạng vân tay hoàn chỉnh được xây dựng bằng Python và OpenCV, với giao diện người dùng thân thiện và hệ thống cơ sở dữ liệu MySQL tích hợp.

## Mục tiêu dự án

Xây dựng một hệ thống có khả năng:
- Chuyển ảnh gốc sang ảnh xám
- Chuẩn hóa và tăng cường ảnh (lọc nhiễu, Gabor filter)
- Nhị phân hóa và làm mảnh ảnh vân tay
- **Trích chọn đặc trưng với 5 phương pháp:**
  - Minutiae chi tiết (ending, bifurcation)
  - LBP (Local Binary Pattern)
  - Ridge Orientation Field
  - Frequency Domain Analysis
  - Feature Matching (SIFT/ORB fallback)
- **So khớp 2 mẫu vân tay với 5 phương pháp:**
  - Minutiae Matching (so khớp minutiae points)
  - Feature Matching (SIFT/ORB features)
  - LBP Texture Matching (so sánh histogram)
  - Ridge Orientation Matching (so sánh hướng sọc)
  - Frequency Domain Matching (phân tích tần số)
- **Lưu trữ người dùng và vân tay trong MySQL database (xla_vantay)**
- **Nhận dạng người dùng từ ảnh vân tay**
- **Hiển thị thông tin người dùng tương ứng**
- Hiển thị các bước xử lý qua giao diện người dùng
- Xuất file kết quả (ảnh + thông số)

## 🏗️ Cấu trúc thư mục

```
XuLyAnh-VanTayNe/
│
├── data/                              # Thư mục lưu ảnh đầu vào
│   └── .gitkeep
│
├── database/                          # Database MySQL
│   └── schema.sql                     # File tạo database
│
├── src/
│   ├── giao_dien/
│   │   ├── __init__.py
│   │   ├── giao_dien_chinh.py         # Giao diện chính Tkinter
│   │   ├── xu_ly_su_kien.py           # Xử lý sự kiện
│   │   ├── hien_thi_ket_qua.py        # Hiển thị kết quả
│   │   └── database_handler.py        # Xử lý sự kiện database
│   │
│   ├── tien_xu_ly/
│   │   ├── __init__.py
│   │   ├── chuyen_xam.py              # Chuyển sang grayscale
│   │   ├── chuan_hoa.py               # Chuẩn hóa ảnh
│   │   ├── loc_nhieu.py               # Lọc nhiễu
│   │   └── tang_cuong.py              # Tăng cường ảnh (Gabor)
│   │
│   ├── phan_doan/
│   │   ├── __init__.py
│   │   └── nhi_phan_hoa.py            # Nhị phân hóa
│   │
│   ├── lam_manh/
│   │   ├── __init__.py
│   │   └── lam_manh_anh.py            # Làm mảnh ảnh (Scikit-image)
│   │
│   ├── trich_dac_trung/
│   │   ├── __init__.py
│   │   ├── trich_dac_trung_chi_tiet.py  # Trích đặc trưng (6 phương pháp)
│   │   └── ve_dac_trung.py              # Vẽ đặc trưng
│   │
│   ├── so_khop/
│   │   ├── __init__.py
│   │   └── so_khop_van_tay.py           # So khớp vân tay (7 phương pháp)
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── database_manager.py          # Quản lý MySQL database
│   │
│   ├── nhan_dang/
│   │   ├── __init__.py
│   │   └── fingerprint_recognition.py   # Nhận dạng người dùng
│   │
│   └── chuong_trinh_chinh.py          # Chương trình main
│
├── ket_qua/                           # Thư mục lưu kết quả
│   └── .gitkeep
│
├── thu_vien_can_thiet.txt             # Danh sách thư viện cần cài
└── README.md                          # File này
```

## Công nghệ sử dụng

- **Python 3.7+**
- **OpenCV (cv2)** - Xử lý ảnh
- **NumPy** - Tính toán số học
- **Scikit-image** - Xử lý ảnh nâng cao
- **SciPy** - Xử lý khoa học
- **Tkinter** - Giao diện người dùng
- **Pillow (PIL)** - Xử lý ảnh PIL
- **MySQL Connector** - Kết nối MySQL database
- **MySQL Workbench** - Quản lý database (tùy chọn)

## Cài đặt

### 1. Cài đặt Python
Đảm bảo bạn đã cài đặt Python 3.7 hoặc cao hơn.

### 2. Cài đặt MySQL Server
- Tải và cài đặt MySQL Server từ [mysql.com](https://www.mysql.com/downloads/)
- Hoặc sử dụng XAMPP/WAMP nếu muốn

### 3. Cài đặt thư viện Python
```bash
python -m pip install -r thu_vien_can_thiet.txt
```

Hoặc cài đặt thủ công:
```bash
pip install opencv-python numpy scikit-image scipy pillow mysql-connector-python
```

### 4. Tạo Database
```bash
# Mở MySQL Command Line hoặc MySQL Workbench
mysql -u root -p

# Chạy file schema.sql
source database/schema.sql

# Hoặc copy toàn bộ nội dung file schema.sql và paste vào MySQL
```

## Hướng dẫn sử dụng

### 1. Chạy chương trình
```bash
python src/chuong_trinh_chinh.py
```

### 2. Các bước xử lý ảnh

#### Bước 1: Chọn ảnh
- Click nút "Chọn ảnh 1" hoặc "Chọn ảnh 2"
- Chọn file ảnh vân tay (.jpg, .png, .bmp)

#### Bước 2: Tiền xử lý
- Click nút "Tiền xử lý"
- Hệ thống sẽ:
  - Chuyển ảnh sang xám
  - Chuẩn hóa ảnh (CLAHE)
  - Lọc nhiễu (Bilateral filter)
  - Tăng cường ảnh (Gabor filter)

#### Bước 3: Nhị phân hóa
- Click nút "Nhị phân hóa"
- Sử dụng phương pháp Otsu tự động tìm ngưỡng

#### Bước 4: Làm mảnh ảnh
- Click nút "Làm mảnh"
- Sử dụng thuật toán Zhang-Suen
- Tự động loại bỏ nhiễu nhỏ

#### Bước 5: Trích chọn đặc trưng
- Click nút "Trích đặc trưng"
- **5 phương pháp trích đặc trưng:**
  1. **Minutiae** - Crossing Number (ending, bifurcation)
  2. **LBP** - Local Binary Pattern (texture)
  3. **Ridge Orientation** - Phân tích hướng đuôi
  4. **Frequency Domain** - Phân tích tần số
  5. **Feature Matching** - SIFT hoặc ORB features

#### Bước 6: So khớp
- Click dropdown "So khớp" để chọn phương pháp
- **5 tùy chọn:**
  1. **Minutiae Matching** - So khớp minutiae points
  2. **Feature Matching** - SIFT/ORB features
  3. **LBP Matching** - LBP histogram comparison
  4. **Ridge Matching** - Ridge orientation fields
  5. **Frequency Matching** - Frequency domain characteristics

- Click nút "Thực hiện"
- Xem kết quả so khớp
- **Chú ý**: Nếu "Khớp" thấp (<10%) nhưng "Tương đồng" cao (>70%), hệ thống sẽ hiển thị dòng cảnh báo: "Cảnh báo: Khớp thấp nhưng tương đồng cao - 2 ảnh có cơ cấu tương tự nhưng có thể khác nhau"

#### Bước 7 (Tùy chọn): Nhận dạng từ Database
- Trước tiên phải kết nối MySQL database (xla_vantay)
- Sử dụng tab "Tìm Kiếm Người Dùng"
- Chọn ảnh vân tay và phương pháp so khớp
- Hệ thống tự động tìm người dùng tương ứng
- Hiển thị thông tin người dùng nếu tìm thấy

#### Bước 8 (Tùy chọn): Đăng ký người dùng mới
- Sử dụng tab "Đăng Ký Người Dùng"
- Nhập thông tin: Username, Họ tên, Email, Số ĐT, CCCD, Chức vụ, Phòng ban
- Chọn ảnh vân tay
- Tự động xử lý ảnh và trích đặc trưng
- Click "Đăng Ký" để lưu vào database

## Thông số kỹ thuật

### Tiền xử lý
- **CLAHE**: clipLimit=2.0, tileGridSize=(8,8)
- **Bilateral Filter**: diameter=9, sigma_color=75, sigma_space=75
- **Gabor Filter**: 6 hướng, kernel_size=21

### Nhị phân hóa
- **Phương pháp**: Otsu's method (tự động)

### Làm mảnh
- **Thuật toán**: Zhang-Suen
- **Lọc noise**: Loại bỏ đường dài < 3 pixels

### Trích chọn đặc trưng - 5 phương pháp

#### 1. Minutiae Features
- **Phương pháp**: Crossing Number
- **Loại điểm**:
  - **Ending**: CN = 1
  - **Bifurcation**: CN = 3
- **Lọc**: Loại bỏ điểm cách nhau < 5 pixels

#### 2. LBP (Local Binary Pattern)
- Texture analysis
- Mỗi pixel so sánh với 8 hàng xóm
- Histogram đặc tính

#### 3. Ridge Orientation Field
- Tính toán hướng ridge tại mỗi điểm
- Gradient-based method
- Consistency measurement

#### 4. Frequency Domain Features
- FFT analysis
- Ridge frequency extraction
- Energy characteristics

#### 5. Feature Matching
- SIFT hoặc ORB features
- Keypoint detection
- Descriptor matching

### So khớp - 5 phương pháp

#### 1. Minutiae Matching
- Khoảng cách tối đa: 50 pixels
- Độ chịu lệch hướng: ±30 độ
- Dựa trên vị trí và hướng minutiae
- Tính "Khớp %" dựa trên số minutiae match

#### 2. Feature Matching
- Phát hiện SIFT hoặc ORB features

- **Cải tiến**: Tính similarity từ `good_matches / total_keypoints` thay vì `good_matches / min_keypoints`
- Cách này công bằng hơn khi 2 ảnh có số keypoints khác biệt lớn

#### 3. LBP Texture Matching
- LBP histogram comparison
- Chi-square distance
- **Cải tiến**: Sử dụng exponential decay `exp(-chi_square/2)` thay vì `100/(1+chi_square)`
- Hạn chế điểm cao khi chi_square nhỏ do trùng hợp

#### 4. Ridge Orientation Matching
- So sánh orientation fields từ cả 2 ảnh
- Mean orientation difference
- **Cải tiến**: Sử dụng exponential decay `exp(-mean_diff/45)` thay vì linear
- Tránh cho điểm cao khi góc khác nhau chỉ 5-10 độ

#### 5. Frequency Domain Matching
- FFT analysis
- Ridge frequency similarity
- Energy similarity
- **Cải tiến**: Sử dụng exponential decay cho từng thành phần
- Chỉ cho điểm cao nếu **TẤT CẢ** đặc trưng tần số tương đồng

### Cảnh báo Consistency
- **Điều kiện**: Khớp < 10% nhưng Tương đồng > 70%
- **Ý nghĩa**: 2 ảnh có cơ cấu tương tự nhưng có thể là vân tay của 2 người khác nhau
- **Hành động**: Hiển thị dòng cảnh báo màu cam giúp người dùng nhận biết

### Database MySQL

#### Table: users
- user_id, username, full_name
- email, phone, address
- date_of_birth, gender
- identification_number
- position, department
- status (active/inactive)

#### Table: fingerprints
- fingerprint_id, user_id
- finger_name (Thumb, Index, Middle, Ring, Pinky)
- hand (Left/Right)
- image_path, image_data (binary)
- minutiae_data (JSON)
- quality_score
- status (approved/pending/rejected)

#### Table: matching_history
- match_id, user_id, fingerprint_id
- query_image_path
- matching_method
- similarity_score
- is_match (true/false)
- matched_at (timestamp)

#### Table: system_settings
- setting_key, setting_value
- Lưu các ngưỡng và cài đặt hệ thống

## 🎨 Giao diện người dùng

Giao diện Tkinter với 3 phần chính:

### 1. Thanh công cụ
- Các nút nhanh để thực hiện các chức năng
- Menu File, Xử lý, Trợ giúp

### 2. Vùng hiển thị ảnh
- Ảnh gốc
- Ảnh sau xử lý
- Ảnh minutiae (với các điểm được vẽ)

### 3. Vùng thông tin
- Kích thước ảnh
- Số ending points
- Số bifurcation points
- Tổng minutiae
- Tỉ lệ so khớp

## 💡 Các hàm chính

### chuyen_xam.py
```python
chuyen_nh_xam(duong_dan_anh)  # Chuyển sang xám từ file
chuyen_xam_tu_mang(anh_goc)   # Chuyển sang xám từ mảng
```

### chuan_hoa.py
```python
chuan_hoa_anh(anh_xam)        # CLAHE
chuan_hoa_tuyến_tính(anh_xam) # Linear normalization
chuan_hoa_z_score(anh_xam)    # Z-score normalization
```

### loc_nhieu.py
```python
loc_nhieu_median(anh_xam)            # Median blur
loc_nhieu_bilateral(anh_xam)         # Bilateral filter
loc_nhieu_gaussian(anh_xam)          # Gaussian blur
loc_nhieu_morphological(anh_xam)     # Morphological operations
```

### tang_cuong.py
```python
ap_dung_gabor_filter(anh_xam)        # Gabor filter
tang_cuong_anh_histogram(anh_xam)    # Histogram equalization
tang_cuong_unsharp_mask(anh_xam)     # Unsharp mask
```

### nhi_phan_hoa.py
```python
nhi_phan_hoa_otsu(anh_xam)           # Otsu's method
nhi_phan_hoa_adaptive(anh_xam)       # Adaptive threshold
nhi_phan_hoa_custom(anh_xam)         # Custom threshold
```

### lam_manh_anh.py
```python
lam_manh_zhang_suen(anh_nhi_phan)    # Zhang-Suen algorithm
lam_manh_scikit_image(anh_nhi_phan)  # Scikit-image method
loc_nhieu_sau_lam_manh(anh_manh)     # Clean skeleton
```

### trich_minhut.py
```python
tinh_crossing_number(anh_manh, i, j)        # Calculate CN at point
phan_loai_minutiae(anh_manh)                # Classify ending/bifurcation
tinh_huong_minutiae(anh_manh, point)        # Calculate orientation
trich_minutiae_chi_tiet(anh_manh)           # Full minutiae extraction
```

### so_khop_van_tay.py
```python
# Phương pháp cơ bản
so_khop_minutiae(minutiae1, minutiae2)           # So khớp minutiae
tinh_diem_tuong_dong_tien_tien(m1, m2)          # Điểm nâng cao
phan_loai_match(score, percentage)              # Phân loại

# Phương pháp chính sử dụng (5 phương pháp)
so_khop_feature_matching(anh1, anh2)            # Feature Matching
so_khop_lbp_texture(anh1, anh2)                 # LBP Texture
so_khop_ridge_orientation(anh1, anh2)           # Ridge Orientation
so_khop_frequency_domain(anh1, anh2)            # Frequency Domain
```

### database_manager.py
```python
# User operations
db.add_user(username, full_name, ...)           # Thêm người dùng
db.get_user_by_id(user_id)                      # Lấy thông tin
db.get_all_users(status='active')               # Danh sách người dùng
db.update_user(user_id, **kwargs)               # Cập nhật
db.delete_user(user_id)                         # Xóa

# Fingerprint operations
db.add_fingerprint(user_id, finger_name, ...)   # Thêm vân tay
db.get_fingerprints_by_user(user_id)            # Lấy vân tay của user
db.get_all_fingerprints(status='approved')      # Tất cả vân tay

# Search & Statistics
db.search_users(keyword)                        # Tìm kiếm
db.get_fingerprints_for_matching()              # Lấy vân tay để match
db.get_statistics()                             # Thống kê
```

### fingerprint_recognition.py
```python
# Nhận dạng
recognition.identify_user_from_minutiae(minutiae, max_results=5)
recognition.identify_user_from_image(anh, minutiae, method='comprehensive')
recognition.get_user_info(user_id)
recognition.save_match_record(...)
```

## 📝 Ví dụ sử dụng lập trình

### Ví dụ 1: So khớp ảnh với 5 phương pháp chính
```python
from src.so_khop.so_khop_van_tay import (
    so_khop_feature_matching,
    so_khop_lbp_texture,
    so_khop_ridge_orientation,
    so_khop_frequency_domain,
    so_khop_minutiae
)

# Minutiae Matching
result = so_khop_minutiae(minutiae1, minutiae2)
print(f"Minutiae: {result['match_percentage']:.2f}%")

# Feature Matching
result = so_khop_feature_matching(anh1, anh2)
print(f"Features: {result['similarity_score']:.2f}")

# LBP Texture
from src.so_khop.so_khop_van_tay import so_khop_lbp_texture
result = so_khop_lbp_texture(anh1, anh2)
print(f"LBP: {result['similarity_score']:.2f}")

# Ridge Orientation
result = so_khop_ridge_orientation(anh1, anh2)
print(f"Ridge: {result['similarity_score']:.2f}")

# Frequency Domain
result = so_khop_frequency_domain(anh1, anh2)
print(f"Frequency: {result['similarity_score']:.2f}")
```

### Ví dụ 2: Làm việc với Database MySQL
```python
from src.database.database_manager import DatabaseManager
from src.nhan_dang.fingerprint_recognition import FingerprintRecognition

# Kết nối database
db = DatabaseManager(host='localhost', user='root', password='123456', 
                     database='xla_vantay')
db.connect()

# Thêm người dùng mới
user_id = db.add_user(
    username='nguyen_van_a',
    full_name='Nguyễn Văn A',
    email='a@example.com',
    phone='0123456789',
    identification_number='123456789'
)

# Lưu vân tay
fingerprint_id = db.add_fingerprint(
    user_id=user_id,
    finger_name='Thumb',
    hand='Right',
    image_path='path/to/image.png',
    minutiae_data=minutiae_dict,
    quality_score=85.5
)

print(f"Lưu thành công! User ID: {user_id}, Fingerprint ID: {fingerprint_id}")

# Ngắt kết nối
db.disconnect()
```

### Ví dụ 3: Nhận dạng người dùng từ vân tay
```python
from src.database.database_manager import DatabaseManager
from src.nhan_dang.fingerprint_recognition import FingerprintRecognition

# Kết nối database
db = DatabaseManager()
db.connect()

# Tạo instance nhận dạng
recognition = FingerprintRecognition(db)

# Thiết lập ngưỡng
recognition.set_threshold(70.0)

# Nhận dạng từ ảnh
results = recognition.identify_user_from_image(
    image=anh_manh,
    minutiae=minutiae_data,
    matching_method='comprehensive',
    max_results=5
)

# Hiển thị kết quả
if results:
    print("Tìm thấy những người dùng tương ứng:")
    for result in results:
        print(f"  - {result['full_name']} ({result['username']})")
        print(f"    Điểm: {result['similarity_score']:.2f}")
        print(f"    Ngón tay: {result['finger_name']}")
else:
    print("Không tìm thấy người dùng tương ứng")

db.disconnect()
```

### Ví dụ 4: Luồng xử lý ảnh hoàn chỉnh
```python
from tien_xu_ly.chuyen_xam import chuyen_nh_xam
from tien_xu_ly.chuan_hoa import chuan_hoa_anh
from tien_xu_ly.tang_cuong import ap_dung_gabor_filter
from phan_doan.nhi_phan_hoa import nhi_phan_hoa_otsu
from lam_manh.lam_manh_anh import lam_manh_scikit_image
from trich_dac_trung.trich_minhut import trich_minutiae_chi_tiet
from so_khop.so_khop_van_tay import so_khop_thong_ke_toan_bo

# 1. Tải và chuyển ảnh
anh_goc, anh_xam = chuyen_nh_xam("fingerprint.jpg")

# 2. Chuẩn hóa
anh_chuan_hoa = chuan_hoa_anh(anh_xam)

# 3. Tăng cường
anh_tang_cuong = ap_dung_gabor_filter(anh_chuan_hoa)

# 4. Nhị phân hóa
anh_nhi_phan, _ = nhi_phan_hoa_otsu(anh_tang_cuong)

# 5. Làm mảnh
anh_manh = lam_manh_scikit_image(anh_nhi_phan)

# 6. Trích chọn đặc trưng
minutiae = trich_minutiae_chi_tiet(anh_manh)

# 7. So khớp (6 phương pháp)
results = so_khop_thong_ke_toan_bo(minutiae1, minutiae2, anh_manh, anh_manh_2)
print(f"Điểm trung bình: {results['overall_score']:.2f}")
print(f"Điểm cao nhất: {results['max_score']:.2f}")
print(f"Minutiae: {results['minutiae_matching']['similarity_score']:.2f}")
print(f"Template: {results['template_matching']['similarity_score']:.2f}")
```

## 🐛 Xử lý lỗi

### Lỗi: "Không thể đọc ảnh"
- Kiểm tra đường dẫn file
- Đảm bảo file tồn tại và có quyền đọc

### Lỗi: "Vui lòng thực hiện tiền xử lý trước"
- Bạn phải hoàn thành các bước theo trình tự

### Lỗi: ImportError
- Cài đặt lại các thư viện: `pip install -r thu_vien_can_thiet.txt`

## 📈 Kế hoạch phát triển

- [x] Minutiae-based matching
- [x] 6 phương pháp so khớp mới (Template, SSIM, Contour, Histogram, Feature, Comprehensive)
- [x] Hệ thống database MySQL
- [x] Lưu trữ người dùng và vân tay
- [x] Nhận dạng người dùng từ vân tay
- [ ] Hỗ trợ webcam real-time
- [ ] Giao diện quản lý người dùng
- [ ] Tối ưu hiệu suất (xử lý nhanh hơn)
- [ ] Export báo cáo chi tiết (PDF/Excel)
- [ ] Hỗ trợ nhập dữ liệu từ scanner
- [ ] API REST cho tích hợp bên thứ ba
- [ ] Mobile app

## 📞 Liên hệ & Hỗ trợ

Nếu gặp vấn đề hoặc có đề xuất, vui lòng liên hệ hoặc tạo issue.

## 📄 Giấy phép

Dự án này được sử dụng cho mục đích giáo dục và nghiên cứu.

## 👥 Tác giả

Dự án nhận dạng vân tay Python-OpenCV

---

**Phiên bản**: 3.0  
**Cập nhật lần cuối**: Tháng 1, 2026  
**Trạng thái**: Hoàn thiện và thêm hệ thống database
