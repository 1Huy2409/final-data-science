# Final Data Science - Phân Cụm Việc Làm Việt Nam

Đề tài: **Phân cụm bộ dữ liệu việc làm Việt Nam**

Dataset: `tinixai/vietnamese-job-descriptions`

Mục tiêu của project là xây dựng pipeline khoa học dữ liệu để:

- Thu thập và mô tả dataset Tinix Vietnam Job Description.
- Khám phá raw data trước khi xử lý.
- Parse salary để tạo `salary_expected_million_vnd`.
- Làm sạch dữ liệu và chia 90% train / 10% test theo yêu cầu môn học.
- Feature engineering từ text, categorical và numeric features.
- Huấn luyện thuật toán phân cụm.
- Đánh giá chất lượng phân cụm bằng các metrics phổ biến.
- Mô tả đặc điểm chung của từng cụm và tự gán nhãn cụm.

## 1. Yêu Cầu Đề Bài

Đề bài mới:

```text
Phân cụm bộ dữ liệu việc làm Việt Nam.

Output:
- lương kỳ vọng = lương trung bình
- số cụm và hiệu suất phân cụm của thuật toán
- các thuộc tính chung/phổ biến của các phần tử thuộc mỗi cụm
- nhãn tự gán cho mỗi cụm

Input:
- toàn bộ dataset Tinix Vietnam Job Description
```

Yêu cầu split theo mẫu tiểu luận:

- Tập huấn luyện: 90% dataset, dùng để huấn luyện và lựa chọn mô hình.
- Tập kiểm thử: 10% dataset, không trùng tập huấn luyện.
- Với bài toán phân cụm, chọn ngẫu nhiên 10 mẫu trong tập kiểm thử để demo kết quả thuật toán phân cụm.

## 2. Dataset

Nguồn dữ liệu:

- Hugging Face: `tinixai/vietnamese-job-descriptions`
- Link: `https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions`

Cách load cơ bản:

```python
from datasets import load_dataset

ds = load_dataset("tinixai/vietnamese-job-descriptions")
df = ds["train"].to_pandas()
```

Các cột chính được dùng trong project:

- `salary`
- `job_title`
- `company_name`
- `location`
- `job_type`
- `job_industry`
- `experience_level`
- `education_level`
- `job_position`
- `job_description`
- `benefits`
- `requirements`
- `year`

## 3. Cấu Trúc Project

```text
Final-Data-Science/
├── notebooks/
│   ├── 01_raw_data_eda.ipynb
│   ├── 02_data_processing.ipynb
│   └── 03_eda_distribution_shift_feature_engineering.ipynb
├── artifacts/
│   ├── raw/
│   ├── clean/
│   ├── audit/
│   └── figures/
├── data_processing.py
├── README.md
├── README_DATA_PROCESSING.md
└── README_EDA.md
```

Vai trò các file chính:

- [01_raw_data_eda.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/01_raw_data_eda.ipynb): tải dataset và EDA trên raw data.
- [02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb): notebook self-contained cho toàn bộ data processing.
- [03_eda_distribution_shift_feature_engineering.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/03_eda_distribution_shift_feature_engineering.ipynb): EDA sau clean, distribution shift và feature engineering. File này cần tiếp tục refactor theo hướng clustering ở phase sau.
- [README_DATA_PROCESSING.md](/D:/H-Coding/Final-Data-Science/README_DATA_PROCESSING.md): mô tả chi tiết phần raw data + processing.
- [data_processing.py](/D:/H-Coding/Final-Data-Science/data_processing.py): bản backup/CLI, không phải file nộp chính.

## 4. Quy Trình Tổng Thể

Pipeline end-to-end của đề tài:

```text
Load dataset
→ Raw EDA
→ Parse salary và tạo salary_expected_million_vnd
→ Lọc dòng salary không hợp lệ
→ Audit outlier
→ Deduplicate
→ Clean text/categorical
→ Split 90/10
→ Feature engineering
→ Clustering
→ Cluster profiling
→ Gán nhãn cụm
→ Báo cáo và slide
```

## 5. Phase 1 - Raw Data Và Data Processing

Phase này đã được triển khai trong [02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb).

Các bước chính:

- Load full dataset từ Hugging Face hoặc cache local.
- Chuẩn hóa và parse `salary`.
- Tạo feature `salary_expected_million_vnd`.
- Chỉ giữ các dòng salary parse hợp lệ.
- Gắn cờ salary outlier, không xóa mặc định.
- Deduplicate theo `id` và full row.
- Audit near-duplicate theo `job_title + company_name + location + salary`.
- Clean text/categorical cơ bản.
- Không remove salary mention mặc định vì đây là bài toán phân cụm, không phải supervised prediction.
- Split 90% train / 10% test.
- Export raw/clean train/test và audit files.

File output chính:

- [cluster_train_raw.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/cluster_train_raw.csv)
- [cluster_test_raw.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/cluster_test_raw.csv)
- [cluster_train_clean.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/cluster_train_clean.csv)
- [cluster_test_clean.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/cluster_test_clean.csv)

Alias giữ để nộp theo format cũ nếu cần:

- [raw_data_train.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/raw_data_train.csv)
- [raw_data_test.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/raw_data_test.csv)
- [clean_data_train.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/clean_data_train.csv)
- [clean_data_test.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/clean_data_test.csv)

Chi tiết đầy đủ nằm trong [README_DATA_PROCESSING.md](/D:/H-Coding/Final-Data-Science/README_DATA_PROCESSING.md).

## 6. Phase 2 - Feature Engineering

Phase feature engineering sẽ dùng `cluster_train_clean.csv` và `cluster_test_clean.csv`.

Feature khuyến nghị:

- Numeric: `salary_expected_million_vnd`, `salary_min`, `salary_max`, `salary_range_width`, `is_salary_outlier`.
- Categorical: `location`, `job_type`, `job_industry`, `experience_level`, `education_level`, `job_position`.
- Text: `job_title`, `job_description`, `requirements`, `benefits`, `company_name`.

Hướng feature engineering:

- Tạo `final_text` từ các trường text/categorical quan trọng.
- TF-IDF word n-gram cho text.
- Có thể thêm SVD để giảm chiều TF-IDF.
- Encode categorical features.
- Scale numeric features.
- Ghép text vector + categorical vector + numeric vector thành feature matrix cuối.

Lưu ý:

- Fit vectorizer/encoder/scaler trên train.
- Transform test bằng object đã fit từ train.
- Không fit lại trên test.

## 7. Phase 3 - Clustering

Các thuật toán clustering có thể dùng:

- K-Means hoặc MiniBatchKMeans.
- Agglomerative Clustering nếu sample/feature size cho phép.
- HDBSCAN hoặc DBSCAN nếu muốn thử density-based clustering.

Quy trình modeling:

1. Fit clustering trên train features.
2. Thử nhiều số cụm `k` nếu dùng K-Means.
3. Đánh giá bằng metrics phổ biến.
4. Chọn mô hình/số cụm hợp lý.
5. Gán cluster label cho train.
6. Với test, dùng model đã fit để predict cluster nếu thuật toán hỗ trợ, hoặc dùng chiến lược nearest centroid.
7. Chọn 10 mẫu test để demo kết quả phân cụm.

Metrics nên dùng:

- Silhouette Score.
- Davies-Bouldin Index.
- Calinski-Harabasz Index.
- Inertia hoặc elbow curve nếu dùng K-Means.

## 8. Phase 4 - Cluster Profiling Và Gán Nhãn Cụm

Sau khi có cluster labels, cần mô tả từng cụm:

- Số lượng mẫu trong cụm.
- Salary mean/median.
- Top `job_industry`.
- Top `location`.
- Top `experience_level`.
- Top `job_type`.
- Top keywords hoặc top terms trong text.
- Một vài job title đại diện.

Từ các đặc điểm phổ biến, nhóm tự gán nhãn cụm.

Ví dụ nhãn có thể là:

- `IT/Software - Mid salary`
- `Sales/Business - Entry level`
- `Management - High salary`
- `Manufacturing/Operations`
- `Intern/Part-time jobs`

Tên cụm cuối cùng phải dựa trên kết quả thật sau khi chạy clustering, không đặt trước theo cảm tính.

## 9. Cách Chạy Project

### Bước 1. Raw EDA

Mở [01_raw_data_eda.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/01_raw_data_eda.ipynb) và chạy từ trên xuống.

Notebook này tạo:

- raw schema summary
- missing summary
- salary pattern summary
- duplicate summary
- top values
- raw EDA figures

### Bước 2. Data Processing

Mở [02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb).

Cell đầu tiên cài thư viện:

```python
%pip install -q pandas numpy pyarrow matplotlib seaborn datasets scikit-learn
```

Khuyến nghị lần đầu:

```python
sample_size=5000
```

Sau khi chạy ổn, đổi lại:

```python
sample_size=None
```

Rồi bấm **Run All** để xử lý full dataset.

Cell quan trọng nhất:

```python
results = run_processing_pipeline(config)
```

Cell này mới thật sự tạo các file CSV output.

### Bước 3. Feature Engineering

Dùng [03_eda_distribution_shift_feature_engineering.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/03_eda_distribution_shift_feature_engineering.ipynb) làm nền, nhưng cần refactor tiếp để bám bài toán clustering.

Input chính:

- `artifacts/clean/cluster_train_clean.csv`
- `artifacts/clean/cluster_test_clean.csv`

### Bước 4. Clustering

Phase này sẽ được triển khai tiếp.

Output kỳ vọng:

- feature matrix train/test
- model clustering đã fit
- bảng metrics
- bảng cluster profile
- bảng 10 mẫu test demo
- figure minh họa cụm nếu có giảm chiều 2D

## 10. Output Hiện Tại Đã Verify

Theo artifact hiện tại:

```text
raw_rows                         606,878
salary_valid_rows_before_dedup   587,904
rows_after_dedup                 587,904
train_rows                       529,113
test_rows                         58,791
test_size                            0.1
remove_salary_mentions             False
```

Train/test salary distribution:

```text
train mean salary:   13.706397
test mean salary:    13.647551
train median salary: 12.0
test median salary:  12.0
```

Outlier:

```text
outlier_count: 42,442
Q1: 9.0
Q3: 15.0
P1: 2.5
P99: 40.0
```

Kết luận:

- Split 90/10 đúng yêu cầu.
- Train/test không lệch salary bất thường.
- Salary parser hoạt động ổn trên phần lớn dataset.
- Outlier được audit, chưa xóa.
- Pipeline processing phù hợp với hướng phân cụm.

## 11. Phân Công Gợi Ý

Người 1 - Raw Data + Processing:

- Raw EDA.
- Salary parser.
- Target/feature creation.
- Outlier audit.
- Deduplicate.
- Clean data.
- Split 90/10.
- Export raw/clean train/test.

Người 2 - EDA Sau Clean + Feature Engineering:

- Distribution shift train/test.
- EDA trên clean data.
- Tạo `final_text`.
- TF-IDF/SVD.
- Encode categorical.
- Scale numeric.
- Xuất feature matrix.

Người 3 - Clustering + Evaluation:

- Train clustering models.
- Chọn số cụm.
- Tính metrics.
- Gán cluster labels.
- Cluster profiling.
- Chọn 10 mẫu test demo.
- Chuẩn bị bảng/figure cho slide.

## 12. Checklist Trước Khi Nộp

Folder nộp bài cần có:

- File PDF báo cáo.
- File PDF slide.
- Notebook source code.
- README hướng dẫn chạy.
- Raw data train/test.
- Clean data train/test.
- Audit/figures nếu nhóm muốn nộp kèm để dễ kiểm chứng.

Kiểm tra bắt buộc:

- PDF báo cáo giống 100% bản in.
- Notebook chạy được theo thứ tự.
- Output files không thiếu.
- Tên đề tài trong folder/đăng ký khớp nội dung báo cáo.
- Không sửa bài sau hạn nộp.

## 13. Ghi Chú Quan Trọng

- `salary_expected_million_vnd` là feature, không phải target supervised prediction.
- Test set trong bài phân cụm dùng để demo/kiểm tra kết quả, không dùng để fit model.
- Nếu thuật toán clustering không có `.predict()`, cần dùng cách gán test vào cụm gần nhất hoặc trình bày rõ cách demo.
- Không xóa outlier mặc định nếu chưa có lý do rõ trong báo cáo.
- Không remove salary mention mặc định trong text ở phase processing.
- Các quyết định cleaning cần có audit để giải thích trong tiểu luận.

