# Báo cáo Notebook 03: EDA, Distribution Shift & Feature Engineering

Notebook 03 chuẩn bị dữ liệu đầu vào cho bài toán **phân cụm việc làm Việt Nam**. Phạm vi của notebook này là: hiểu dữ liệu sạch sau preprocessing, kiểm tra train/test distribution shift, tạo feature text-only không rò rỉ nhãn ngành, giảm chiều và xuất artifact để người làm clustering dùng tiếp.

Notebook này **không chọn số cụm và không tạo kết quả clustering cuối cùng**. Cell MiniBatchKMeans cuối notebook chỉ là sanity check để kiểm tra embedding có thể phân cụm, không phải model chính thức.

## 1. EDA Dữ Liệu Sạch

Notebook đọc dữ liệu từ `artifacts/clean/clean_data_train.csv` và `artifacts/clean/clean_data_test.csv`, mặc định với `SAMPLE_SIZE = 20000` để phát triển nhanh. Khi cần artifact cuối trên toàn bộ dữ liệu, đổi `SAMPLE_SIZE = None` và chạy lại notebook.

Các kiểm tra chính:

- Schema, kiểu dữ liệu, missing value và số dòng train/test.
- Phân phối salary, salary outlier và log salary.
- Độ dài `job_description`, `requirements`, `benefits`, `final_text`.
- Phân phối metadata như `job_industry`, `experience_level`, `job_type`, `education_level`, `job_position`, `location_simplified`.
- Tần suất unigram/ngram để hiểu các từ/cụm từ phổ biến trong JD.

Ý nghĩa với bài toán:

- Salary có phân phối lệch phải và có outlier, nên chỉ giữ để profile cụm thay vì đưa trực tiếp vào feature clustering.
- Text của JD/title/requirements/benefits là nguồn tín hiệu chính để mô hình tự phát hiện nhóm việc làm.
- Metadata categorical rất hữu ích để mô tả và kiểm tra cụm sau này, nhưng không nên dùng làm input clustering vì có thể ép cụm theo nhãn có sẵn.

## 2. Distribution Shift

Notebook so sánh train/test bằng:

- KS test cho biến numeric: `salary_expected_million_vnd`, `log_salary_expected`, `salary_range_width`, `final_text_length`.
- Chênh lệch tỷ lệ cho categorical: `job_type`, `experience_level`, `education_level`, `job_position`, `job_industry`, `location_simplified`.

Kết quả hiện tại cho thấy train/test không lệch đáng kể ở các biến quan trọng. Điều này giúp người làm clustering yên tâm fit pipeline trên train và transform/demo trên test mà không gặp shift lớn.

Lưu ý: trong clustering không có target supervised. Vì vậy `job_industry` không phải nhãn train, mà là metadata dùng để đánh giá ngoại sinh và giải thích cụm.

## 3. Feature Engineering

Pipeline đã được sửa theo hướng **text-only, không leakage**:

```text
job_title + job_description + requirements + benefits
-> TF-IDF
-> TruncatedSVD 300 chiều
-> bỏ SVD component đầu tiên
-> UMAP 10 chiều
```

Các cột được dùng để tạo feature clustering:

- `job_title`
- `job_description`
- `requirements`
- `benefits`

Các cột không đưa vào feature matrix, chỉ giữ làm metadata:

- `job_industry`
- `experience_level`
- `job_type`
- `education_level`
- `job_position`
- `location_simplified`
- `salary_expected_million_vnd`, `salary_min`, `salary_max`, `salary_range_width`
- `is_salary_outlier`
- các text length features

Lý do không dùng `job_industry` và OHE:

- Mục tiêu clustering là để thuật toán tự phát hiện nhóm việc làm từ nội dung JD.
- Nếu đưa `job_industry` vào input, mô hình có thể học lại nhãn ngành có sẵn thay vì học ngữ nghĩa văn bản.
- Không còn `OneHotEncoder`, `OHE_WEIGHT`, hoặc categorical feature trong `X_train_features`.

TF-IDF hiện dùng:

- `max_features=30000`
- `ngram_range=(1, 3)`
- `min_df=20`
- `max_df=0.40`
- `sublinear_tf=True`
- stopwords tiếng Việt đã được token hóa để tránh warning không nhất quán của `TfidfVectorizer`.

## 4. Dimensionality Reduction & Validation

Notebook dùng `TruncatedSVD` để nén TF-IDF sparse xuống 300 chiều, sau đó bỏ component đầu tiên trước khi đưa vào UMAP. Component đầu thường có thể chứa template/common wording, nên loại bỏ giúp embedding tập trung hơn vào tín hiệu phân biệt.

UMAP tạo embedding 10 chiều để bàn giao cho clustering:

- `X_train_umap.npy`
- `X_test_umap.npy`

Biểu đồ UMAP/t-SNE 2D được tô màu theo `job_industry` và `experience_level`, nhưng đây chỉ là **external validation/profiling**. Các metadata này không nằm trong input. Nếu màu ngành có xu hướng gom vùng, đó là tín hiệu text đã học được cấu trúc ngành nghề từ JD.

Sanity check hiện tại:

- `X_train_features`: TF-IDF text-only, shape `(20000, 30000)`.
- `X_train_svd`: shape `(20000, 300)`.
- `X_train_umap`: shape `(20000, 10)`.
- `numeric_features_used_for_clustering = 0`.
- `metadata_features_used_for_clustering = 0`.

## 5. Artifact Bàn Giao

Các file chính cho người làm clustering:

```text
artifacts/features/X_train_umap.npy
artifacts/features/X_test_umap.npy
artifacts/features/train_metadata.csv
artifacts/features/test_metadata.csv
artifacts/features/feature_pipeline.joblib
artifacts/features/tfidf_model.joblib
artifacts/features/svd_model.joblib
artifacts/features/umap_model.joblib
```

Các file phụ hữu ích:

```text
artifacts/features/X_train_features.npz
artifacts/features/X_test_features.npz
artifacts/features/X_train_svd.npy
artifacts/features/X_test_svd.npy
artifacts/features/feature_engineering_summary.csv
artifacts/features/distribution_shift_numeric.csv
artifacts/features/distribution_shift_categorical.csv
```

Ví dụ load dữ liệu:

```python
import numpy as np
import pandas as pd
from scipy import sparse
import joblib

X_train = np.load("artifacts/features/X_train_umap.npy")
X_test = np.load("artifacts/features/X_test_umap.npy")
train_meta = pd.read_csv("artifacts/features/train_metadata.csv")
test_meta = pd.read_csv("artifacts/features/test_metadata.csv")

pipeline = joblib.load("artifacts/features/feature_pipeline.joblib")
tfidf = joblib.load("artifacts/features/tfidf_model.joblib")
svd = joblib.load("artifacts/features/svd_model.joblib")
umap_model = joblib.load("artifacts/features/umap_model.joblib")
```

## 6. Ghi Chú Cho Người Làm Clustering

Người làm clustering nên:

- Fit thuật toán clustering trên `X_train_umap.npy`.
- Dùng `X_test_umap.npy` để demo/kiểm tra gán cụm trên test, không fit lại TF-IDF/SVD/UMAP trên test.
- Thử nhiều cấu hình như KMeans với nhiều `k`, HDBSCAN hoặc Agglomerative nếu phù hợp.
- Báo cáo metrics phổ biến: silhouette, Davies-Bouldin, Calinski-Harabasz; nếu dùng HDBSCAN thì thêm số cụm và noise ratio.
- Gắn `cluster_id` vào `train_metadata.csv`.
- Profile từng cụm bằng metadata: top `job_industry`, top `job_title`, top từ khóa trong JD, `experience_level` phổ biến, salary median, top location/job type.
- Tự đặt nhãn cụm dựa trên thuộc tính phổ biến, ví dụ: Sales/CSKH, Kế toán/Tài chính, Kỹ thuật/Sản xuất, Marketing/Nội dung.

Không nên:

- Fit clustering trực tiếp trên `job_industry`.
- Dùng `job_industry` làm feature đầu vào.
- Diễn giải UMAP/t-SNE tô màu metadata như bằng chứng supervised. Đó chỉ là kiểm tra ngoại sinh sau khi embedding đã được học từ text.

Kết luận: Notebook 03 đã hoàn thành vai trò EDA, distribution shift và feature engineering. Output đã đủ sạch và đủ thông tin để bàn giao cho bước clustering chính thức.
