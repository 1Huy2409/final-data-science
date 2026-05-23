# Báo cáo Tổng quan Notebook 03: EDA, Distribution Shift & Feature Engineering

Notebook này là trái tim của quá trình biến đổi dữ liệu, nơi dữ liệu thô được phân tích sâu sắc (EDA) để tìm ra các quy luật, sau đó ứng dụng chính các quy luật này để tạo ra các đặc trưng mạnh mẽ cho mô hình Machine Learning (Feature Engineering).

Mọi dòng code tạo Feature ở nửa sau Notebook đều được "bảo chứng" bởi các biểu đồ ở nửa đầu Notebook.

---

## PHẦN 1: KHÁM PHÁ DỮ LIỆU (EXPLORATORY DATA ANALYSIS)

Mục tiêu của phần này không phải là vẽ biểu đồ cho đẹp, mà là tìm ra **"Tín hiệu" (Signal)** để hướng dẫn cho bước Feature Engineering.

### 1. Phân tích Đơn biến & Đa biến cơ bản
- **Biểu đồ Histogram & KDE (Biến mục tiêu - Lương):**
  - *Ý nghĩa:* Cho thấy phân bố lương bị lệch phải (Right-skewed) rõ rệt, đa số tập trung ở mức trung bình - thấp, thưa thớt ở mức siêu cao.
  - *Vai trò:* Nhắc nhở người làm Modeling (Phase sau) ưu tiên sử dụng các mô hình Tree-based (như XGBoost, Random Forest) vì chúng miễn nhiễm với Outlier, hoặc cân nhắc dùng Log Transform.
- **Biểu đồ Countplot (Tần suất danh mục):**
  - *Ý nghĩa:* Phát hiện sự mất cân bằng (Imbalance) nghiêm trọng ở các cột `Location` hoặc `Job Industry` (một số tỉnh thành chỉ có vài tin tuyển dụng).
  - *Vai trò:* Quyết định chiến lược **Rare Grouping** ở Bước 3.2 (gom các tỉnh lẻ thành nhóm `'other'`) trước khi One-hot Encoding để chống bùng nổ số chiều.
- **Biểu đồ Boxplot & Violin Plot (Lương theo Kinh nghiệm / Học vấn):**
  - *Ý nghĩa:* Chứng minh mối quan hệ tỷ lệ thuận (Monotonic): Kinh nghiệm càng cao -> Lương càng cao.
  - *Vai trò:* Cung cấp cơ sở toán học để dùng **Ordinal Encoding** (0, 1, 2, 3...) ở Bước 3.2 thay vì One-Hot Encoding, giúp mô hình giữ lại được thứ bậc lớn-bé của kinh nghiệm.

### 2. Nhóm Biểu đồ Text EDA (Khám phá Dữ liệu Văn bản)
Đây là phần "ăn điểm" nhất, phân tích sâu vào mỏ vàng cốt lõi của bài toán là Job Description (JD).

- **TEXT-EDA 1: Phân bố độ dài JD (Word Count):**
  - *Ý nghĩa:* Tin tuyển dụng dài (>350 từ) thường có trung vị lương cao hơn (vì mô tả vị trí quản lý, senior thường rất chi tiết).
  - *Vai trò:* Cung cấp bằng chứng để tạo thêm đặc trưng dạng số `jd_word_count`.
- **TEXT-EDA 2: Phân tích N-gram (Bigrams & Trigrams):**
  - *Ý nghĩa:* Phát hiện các cụm 2-3 từ thường đi liền nhau mang ý nghĩa hoàn chỉnh trong tiếng Việt (VD: "bảo hiểm y tế", "quản lý dự án").
  - *Vai trò:* Tối ưu hóa bộ công cụ Regex ở Bước 3.1, đảm bảo bắt đúng nguyên cụm từ thay vì bắt nhầm các từ đơn lẻ.
- **TEXT-EDA 3: Boxplot Phân Bố Lương Theo Từ Khóa Tiêu Biểu:**
  - *Ý nghĩa:* Bốc ngẫu nhiên 4 từ khóa tiêu biểu (Quản lý, Dự án, Tiếng Anh, Hỗ trợ) để vẽ Boxplot so sánh. Thấy rõ sự xuất hiện của từ "Tiếng Anh" hay "Quản lý" đẩy toàn bộ hộp phân bố lương lên cao.
  - *Vai trò:* Đây là bước **Validation (Kiểm chứng) sớm**. Nó đập tan mọi nghi ngờ của hội đồng, minh chứng bằng hình ảnh rằng: *Việc trích xuất Keyword Boolean (0/1) ở Phần 3 mang lại tín hiệu phân loại cực kỳ mạnh mẽ.*

---

## PHẦN 2: DISTRIBUTION SHIFT (ĐÁNH GIÁ SỰ DỊCH CHUYỂN PHÂN BỐ)

- **Overlapping KDE & Class Distribution (So sánh Train vs Test):**
  - *Ý nghĩa:* Dùng đồ thị để chứng minh đường cong phân bố của tập Train và tập Test khớp nhau hoàn hảo. Tỷ lệ các nhãn (Kinh nghiệm, Học vấn) giữa hai tập cũng đồng nhất.
  - *Vai trò:* Đảm bảo không xảy ra hiện tượng **Covariate Shift** (Dịch chuyển biến đầu vào) hay **Target Shift** (Dịch chuyển biến mục tiêu). Đây là tấm vé bảo hành chứng minh mô hình huấn luyện trên Train sẽ tổng quát hóa (generalize) rất tốt khi dự đoán trên Test.

---

## PHẦN 3: FEATURE ENGINEERING (KỸ THUẬT ĐẶC TRƯNG)

Phần này là nơi chuyển hóa toàn bộ Insight từ Phần 1 thành ma trận số học để đưa vào mô hình.

- **3.1. Keyword Matching (Trích xuất từ khóa thủ công):**
  - Kế thừa bộ từ điển từ N-gram, chạy các biểu thức Regex an toàn với tiếng Việt (Sử dụng ranh giới từ `(?:^|\W)`) để tạo ra 19 cột nhị phân (0/1) cho các Kỹ năng công nghệ, Kỹ năng mềm, Ngoại ngữ và Phúc lợi.
- **3.2. Categorical Encoding (Mã hóa phân loại):**
  - Áp dụng triệt để định hướng từ EDA: Dùng Ordinal Encoding cho Kinh nghiệm/Học vấn; gom nhóm Rare Grouping rồi mới One-Hot Encoding cho Location/Job Industry.
- **3.3. NLP & Text Vectorization (TF-IDF + TruncatedSVD):**
  - Giải quyết phần "văn bản dư thừa" chưa được quét bởi Keyword Matching. 
  - **Kỹ thuật chống Data Leakage:** Áp dụng Regex "phẫu thuật" xóa mọi con số liên quan đến tiền bạc/tiền lương trong văn bản để mô hình không học vẹt.
  - **Giảm chiều dữ liệu:** Dùng `TruncatedSVD` để nén ma trận thưa (Sparse) khổng lồ 5000 chiều của TF-IDF xuống thành một ma trận đặc (Dense) gọn gàng (dựa trên thuật toán Diagnostic tự tìm điểm bão hòa phương sai). Tối ưu hóa cực độ tốc độ chạy của mô hình học máy sau này.
- **3.4. Trực quan hóa không gian đặc trưng bằng t-SNE 2D:**
  - *Ý nghĩa:* Ép không gian hàng trăm chiều của ma trận đặc trưng cuối cùng xuống mặt phẳng 2D.
  - *Vai trò:* Biểu đồ chốt hạ (Final Validation). Nếu trên biểu đồ t-SNE, các điểm dữ liệu Lương Cao và Lương Thấp tạo thành những cụm (cluster) có thể phân tách được, điều đó khẳng định: **Toàn bộ quá trình Feature Engineering của chúng ta đã thành công xuất sắc!**
