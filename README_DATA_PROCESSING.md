# Data Processing Guide - Vietnamese Job Clustering

Tài liệu này mô tả toàn bộ phần **Raw Data + Data Processing** cho đề tài mới:

**Phân cụm bộ dữ liệu việc làm Việt Nam**

Dataset sử dụng: `tinixai/vietnamese-job-descriptions`

Mục tiêu của phase này là chuẩn bị dữ liệu đầu vào đủ sạch, đủ audit và đúng yêu cầu môn học để các bước sau có thể làm feature engineering và clustering.

## 1. File Liên Quan

File notebook chính:

- [01_raw_data_eda.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/01_raw_data_eda.ipynb): tải dataset và EDA trên raw data.
- [02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb): notebook self-contained cho toàn bộ data processing.

File backup:

- [data_processing.py](/D:/H-Coding/Final-Data-Science/data_processing.py): bản CLI/backup của pipeline, không phải file nộp chính.

Thư mục output:

- [artifacts/raw](/D:/H-Coding/Final-Data-Science/artifacts/raw): dữ liệu sau parse salary, lọc valid, deduplicate và split.
- [artifacts/clean](/D:/H-Coding/Final-Data-Science/artifacts/clean): dữ liệu đã clean text/categorical, dùng cho feature engineering.
- [artifacts/audit](/D:/H-Coding/Final-Data-Science/artifacts/audit): bảng audit, summary, parser test, outlier.
- [artifacts/figures](/D:/H-Coding/Final-Data-Science/artifacts/figures): biểu đồ phục vụ báo cáo/slide.

## 2. Thay Đổi Theo Đề Bài Mới

Đề ban đầu là **dự đoán lương kỳ vọng**. Đề mới là **phân cụm dữ liệu việc làm Việt Nam**.

Vì vậy pipeline đã đổi theo hướng:

- `salary_expected_million_vnd` không còn là target để dự đoán, mà là **numeric feature quan trọng cho clustering**.
- Vẫn cần parse `salary` vì đề bài yêu cầu output lương kỳ vọng và salary giúp mô tả cụm.
- Vẫn chia **90% train / 10% test** theo yêu cầu môn học.
- Train set dùng để feature engineering, chọn số cụm và fit thuật toán phân cụm.
- Test set dùng để demo/kiểm tra kết quả clustering, trong đó đề yêu cầu chọn ngẫu nhiên 10 mẫu test để trình bày.
- Không xóa salary mention trong text theo mặc định, vì đây không còn là bài toán prediction leakage. Salary mention có thể là tín hiệu ngữ cảnh hữu ích khi phân cụm.

## 3. Nguồn Dữ Liệu Và Cách Thu Thập

Nguồn dữ liệu:

- Hugging Face dataset: `tinixai/vietnamese-job-descriptions`
- Link: `https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions`

Công cụ thu thập:

- Python
- `datasets`
- `pandas`

Cách load dataset chuẩn:

```python
from datasets import load_dataset

ds = load_dataset("tinixai/vietnamese-job-descriptions")
df = ds["train"].to_pandas()
```

Trong notebook/pipeline, dữ liệu được load theo thứ tự ưu tiên:

1. Nếu Hugging Face cache local đã có `data.parquet`, đọc trực tiếp từ cache để chạy nhanh hơn.
2. Nếu chưa có cache, gọi `load_dataset("tinixai/vietnamese-job-descriptions")`.

Đầu vào của bước thu thập:

- Dataset ID trên Hugging Face.

Đầu ra của bước thu thập:

- DataFrame raw chứa toàn bộ job postings.

## 4. Notebook Processing Được Tổ Chức Như Thế Nào

[02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb) hiện đã được tách thành nhiều cell nhỏ, không còn gom toàn bộ code vào một cell lớn.

Các phần chính trong notebook:

- Cài thư viện bằng `%pip install`.
- Imports, hằng số và cấu hình pipeline.
- Load dataset từ Hugging Face hoặc cache local.
- Helper làm sạch text và chuẩn hóa salary string.
- Salary parser và target/feature creation.
- Raw EDA và audit helpers.
- Filtering, outlier audit, deduplicate, cleaning và split.
- Figures, parser tests, audit tables và export helpers.
- Pipeline runner.
- Cell config chạy thực tế.
- Cell test parser.
- Cell run processing và export.
- Cell validate output files.
- Cell preview dữ liệu sạch.

Bạn có thể bấm **Run All** để chạy toàn bộ, nhưng lần đầu nên đổi `sample_size=5000` để test nhanh trước.

## 5. Quy Trình Xử Lý Dữ Liệu

### Bước 1. Raw EDA

Raw EDA được thực hiện trước processing để hiểu chất lượng dữ liệu gốc.

Mục tiêu:

- Xác nhận schema thực tế.
- Thống kê số dòng, số cột, kiểu dữ liệu.
- Thống kê missing value.
- Thống kê duplicate.
- Khảo sát pattern của `salary`.
- Khảo sát các cột categorical chính.
- Khảo sát độ dài text.
- Kiểm tra salary mention trong các trường text.
- Xuất bảng audit và figure phục vụ báo cáo.

Output raw EDA:

- `raw_schema_summary.csv`
- `raw_missing_summary.csv`
- `raw_text_length_summary.csv`
- `raw_salary_pattern_summary.csv`
- `raw_duplicate_summary.csv`
- `raw_top_values.csv`
- `raw_salary_mention_summary.csv`
- `raw_salary_outlier_examples.csv`
- `raw_rule_candidates.json`

### Bước 2. Chuẩn Hóa Chuỗi Salary

Cột `salary` ban đầu là text, có nhiều format khác nhau.

Pipeline chuẩn hóa:

- Lowercase.
- Normalize Unicode.
- Chuẩn hóa whitespace.
- Chuẩn hóa dấu phân cách range như `-`, `~`, `to`.
- Chuẩn hóa đơn vị tiền tệ như `usd`, `$`, `triệu`, `tr`, `vnd`, `vnđ`.

Ví dụ:

- `26 - 36 triệu` được hiểu là khoảng 26 đến 36 triệu VND.
- `4000 usd` được hiểu là 4000 USD.
- `14.000.000 - 20.000.000 vnd` được hiểu là 14 đến 20 triệu VND.

### Bước 3. Phân Loại Salary

Mỗi dòng salary được phân loại thành một trong các nhóm:

- `range_vnd`
- `single_vnd`
- `range_usd`
- `single_usd`
- `range_unknown`
- `single_unknown`
- `ambiguous`
- `invalid`
- `missing`

Ý nghĩa:

- `range_*`: có khoảng lương min-max.
- `single_*`: có một mức lương đơn.
- `unknown`: có số nhưng không xác định được đơn vị.
- `ambiguous`: mơ hồ như `thỏa thuận`, `cạnh tranh`, `negotiable`.
- `invalid`: không parse được thành thông tin lương có nghĩa.

### Bước 4. Tạo Feature Lương Kỳ Vọng

Pipeline tạo các cột:

- `salary_raw_normalized`
- `salary_pattern`
- `salary_currency`
- `salary_min`
- `salary_max`
- `salary_expected_million_vnd`
- `salary_parse_status`
- `salary_range_width`

Công thức:

```text
salary_expected_million_vnd = (salary_min + salary_max) / 2
```

Nếu salary là mức đơn:

```text
salary_min = salary_max
salary_expected_million_vnd = salary_min
```

Quy đổi:

- VND được đổi về triệu VND bằng cách chia `1_000_000`.
- USD được quy đổi theo tỷ giá cố định `25,000 VND/USD`, sau đó đổi sang triệu VND.

Ví dụ parser hiện tại:

- `26 - 36 triệu` -> `31.0`
- `15 - 35 triệu` -> `25.0`
- `4000 usd` -> `100.0`
- `1200 usd` -> `30.0`
- `14.000.000 - 20.000.000 vnd` -> `17.0`
- `thỏa thuận` -> không hợp lệ/ambiguous
- `đang cập nhật` -> invalid

### Bước 5. Lọc Dòng Có Salary Không Hợp Lệ

Vì salary là feature quan trọng của clustering, pipeline v1 chỉ giữ các dòng có salary rõ ràng.

Điều kiện giữ:

- `salary_parse_status == "valid"`
- `salary_expected_million_vnd > 0`
- `salary_min <= salary_max`
- `salary_expected_million_vnd` không null
- `salary_expected_million_vnd` không infinite

Các dòng bị loại:

- salary mơ hồ
- salary invalid
- salary không có đơn vị rõ
- salary null hoặc không tạo được numeric feature hợp lệ

Lý do chọn hướng này:

- Ưu tiên feature salary sạch.
- Giảm nhiễu cho clustering.
- Các dòng salary mơ hồ có thể xử lý ở version sau nếu cần, nhưng không nên trộn vào v1.

### Bước 6. Outlier Audit

Outlier salary không bị xóa mặc định. Pipeline chỉ audit và gắn cờ.

Các thống kê được tính:

- `Q1`
- `Q3`
- `IQR`
- `P1`
- `P99`
- `lower_iqr`
- `upper_iqr`

Cột tạo thêm:

- `is_salary_outlier`
- `outlier_reason`

Rule gắn cờ:

- Nhỏ hơn ngưỡng IQR lower hoặc P1.
- Lớn hơn ngưỡng IQR upper hoặc P99.

Ý nghĩa:

- Người làm feature engineering/clustering có thể thử 2 kịch bản: giữ outlier hoặc loại outlier.
- Báo cáo có thể trình bày outlier như một phần chất lượng dữ liệu.

### Bước 7. Deduplicate

Pipeline xử lý duplicate theo 3 lớp:

- Duplicate theo `id`.
- Duplicate toàn bộ row.
- Near-duplicate theo `job_title + company_name + location + salary`.

Rule hiện tại:

- Drop duplicate theo `id`.
- Drop duplicate toàn bộ row.
- Near-duplicate được xuất audit, không xóa hàng loạt mặc định.

Output:

- `raw_duplicate_summary.csv`
- `near_duplicate_audit.csv`

### Bước 8. Clean Text Và Categorical

Áp dụng cho các cột:

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

Rule cleaning:

- Lowercase.
- Normalize Unicode.
- Bỏ HTML.
- Bỏ URL.
- Bỏ email.
- Bỏ phone.
- Bỏ ký tự lặp vô nghĩa.
- Chuẩn hóa whitespace.
- Chuyển chuỗi rỗng thành `unknown`.

Không làm ở phase Người 1:

- Stemming.
- Stopword removal.
- Tokenization đặc thù.
- TF-IDF.
- One-hot encoding.
- Scaling.
- Clustering model.

Các bước trên thuộc phần Người 2/Người 3.

### Bước 9. Không Remove Salary Mention Mặc Định

Khác với bài toán dự đoán lương, bài toán hiện tại là phân cụm.

Vì vậy:

- `remove_salary_mentions=False` mặc định.
- Text vẫn được clean cơ bản.
- Salary mention không bị xóa hàng loạt.

Lý do:

- Trong clustering, salary mention có thể là tín hiệu giúp mô tả nhóm việc làm.
- Không còn rủi ro target leakage theo nghĩa supervised prediction.

Pipeline vẫn ghi audit:

- `clean_salary_mention_summary.csv`
- `clean_salary_leakage_summary.csv` alias tương thích với tên cũ

### Bước 10. Split 90/10

Split được thực hiện sau khi:

1. Parse salary.
2. Lọc salary hợp lệ.
3. Audit outlier.
4. Deduplicate.
5. Clean text/categorical.

Cấu hình:

- `test_size = 0.1`
- `random_state = 42`
- Nếu có thể, stratify theo bin của `salary_expected_million_vnd`.

Ý nghĩa:

- Train 90%: dùng cho feature engineering, chọn số cụm, fit clustering.
- Test 10%: dùng để demo và kiểm tra kết quả phân cụm.

Lưu ý:

- Đây không phải train/test theo nghĩa supervised prediction.
- Split này tồn tại để bám yêu cầu môn học và để có tập demo không trùng train.

### Bước 11. Export

Pipeline xuất 4 file chính theo naming mới:

- [cluster_train_raw.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/cluster_train_raw.csv)
- [cluster_test_raw.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/cluster_test_raw.csv)
- [cluster_train_clean.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/cluster_train_clean.csv)
- [cluster_test_clean.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/cluster_test_clean.csv)

Đồng thời giữ alias cũ để tương thích yêu cầu nộp:

- [raw_data_train.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/raw_data_train.csv)
- [raw_data_test.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/raw_data_test.csv)
- [clean_data_train.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/clean_data_train.csv)
- [clean_data_test.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/clean_data_test.csv)

Ý nghĩa:

- `cluster_*_raw.csv`: dữ liệu đã có salary feature hợp lệ, đã split, chưa clean text/categorical sâu.
- `cluster_*_clean.csv`: dữ liệu đã clean cơ bản, sẵn sàng cho feature engineering.
- `raw_data_*` và `clean_data_*`: alias để đáp ứng format nộp bài nếu giảng viên yêu cầu tên file này.

## 6. Audit Và Metadata Output

Các file audit quan trọng:

- `processing_summary.csv`: số dòng qua từng bước processing.
- `salary_parse_audit.csv`: thống kê parse status và salary pattern.
- `salary_parser_tests.csv`: kết quả test parser bằng các ví dụ mẫu.
- `salary_outlier_summary.csv`: ngưỡng outlier và số lượng outlier.
- `salary_outlier_examples.csv`: ví dụ outlier để review thủ công.
- `raw_duplicate_summary.csv`: thống kê duplicate raw.
- `near_duplicate_audit.csv`: các dòng near-duplicate.
- `train_test_salary_distribution.csv`: phân phối salary train/test.
- `data_dictionary.csv`: mô tả role các cột cho downstream.

Các figure quan trọng:

- `raw_salary_patterns.png`
- `raw_text_length_boxplot.png`
- `salary_string_length_hist.png`
- `top_job_industry.png`
- `top_location.png`
- `top_job_type.png`
- `top_experience_level.png`
- `top_education_level.png`
- `top_job_position.png`
- `salary_feature_distribution_by_split.png`
- `salary_feature_boxplot.png`
- `outlier_flag_ratio.png`

## 7. Cách Chạy Notebook

Khuyến nghị chạy lần đầu:

1. Mở [02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb).
2. Chạy cell đầu tiên để cài thư viện.
3. Chạy các cell định nghĩa hàm từ trên xuống.
4. Ở cell cấu hình, test nhanh bằng:

```python
sample_size=5000
```

5. Chạy tiếp đến hết notebook.
6. Nếu sample chạy ổn, đổi lại:

```python
sample_size=None
```

7. Bấm **Run All** để xử lý full dataset.

Cell quan trọng nhất để tạo file output là:

```python
results = run_processing_pipeline(config)
```

Nếu chỉ chạy đến trước cell này thì notebook mới định nghĩa hàm, chưa tạo output mới.

## 8. Kết Quả Verify Hiện Tại

Dựa trên artifact hiện tại trong `artifacts/audit`, full processing đã tạo được output với số liệu:

```text
raw_rows                         606,878
salary_valid_rows_before_dedup   587,904
rows_after_dedup                 587,904
train_rows                       529,113
test_rows                         58,791
test_size                            0.1
random_state                          42
remove_salary_mentions             False
```

Salary parse status:

```text
valid              595,397
invalid              7,562
ambiguous            3,408
unknown_currency       510
missing                  1
```

Salary pattern phổ biến:

```text
range_vnd      585,281
single_vnd       9,932
invalid          7,562
ambiguous        3,408
range_unknown      350
single_unknown     153
single_usd          99
range_usd           92
missing              1
```

Train/test salary distribution:

```text
train rows: 529,113
test rows:   58,791
train ratio: 0.899999
test ratio:  0.100001
train mean salary: 13.706397
test mean salary:  13.647551
train median salary: 12.0
test median salary:  12.0
```

Outlier summary:

```text
Q1: 9.0
Q3: 15.0
P1: 2.5
P99: 40.0
IQR: 6.0
lower_iqr: 0.0
upper_iqr: 24.0
outlier_count: 42,442
```

Kết luận verify:

- Split 90/10 hiện tại đúng yêu cầu.
- Train/test có median salary giống nhau và mean gần nhau, không lệch bất thường.
- Parser salary đang hoạt động ổn với phần lớn dataset.
- Các dòng salary không rõ đã được loại khỏi dataset clustering v1.
- Outlier được gắn cờ nhưng chưa xóa.
- `remove_salary_mentions=False`, đúng với hướng bài toán phân cụm.

## 9. Tiêu Chí Kiểm Tra Chất Lượng

Sau khi chạy lại processing, cần kiểm tra:

- `salary_parser_tests.csv` có tất cả test case pass.
- Mọi dòng trong train/test có `salary_parse_status == "valid"`.
- Mọi dòng có `salary_expected_million_vnd > 0`.
- Mọi dòng có `salary_min <= salary_max`.
- Train/test không trùng nhau theo `id`.
- Train/test có cùng schema.
- Tỷ lệ train/test xấp xỉ 90/10.
- `train_test_salary_distribution.csv` không cho thấy lệch bất thường.
- 4 file output chính đọc lại được bằng `pandas`.

## 10. Bàn Giao Cho Feature Engineering Và Clustering

Người 2 có thể dùng:

- `cluster_train_clean.csv`
- `cluster_test_clean.csv`
- `data_dictionary.csv`
- `salary_outlier_summary.csv`
- `train_test_salary_distribution.csv`

Cột numeric khuyến nghị:

- `salary_expected_million_vnd`
- `salary_min`
- `salary_max`
- `salary_range_width`
- `is_salary_outlier`

Cột categorical khuyến nghị:

- `location`
- `job_type`
- `job_industry`
- `experience_level`
- `education_level`
- `job_position`

Cột text khuyến nghị:

- `job_title`
- `job_description`
- `requirements`
- `benefits`
- `company_name`

Gợi ý tạo `final_text` ở phase sau:

```text
job_title + job_description + requirements + benefits + location + job_industry + experience_level
```

Lưu ý cho Người 2/Người 3:

- Không cần fit clustering trên test set.
- Test set dùng để demo/kiểm tra nhãn cụm sau khi model hoặc pipeline clustering đã fit trên train.
- Nếu loại outlier ở phase sau, phải ghi rõ rule và so sánh kết quả trước/sau.

## 11. Tóm Tắt Ngắn Gọn

Pipeline hiện tại đã phù hợp với đề phân cụm mới:

- Dùng toàn bộ dataset Tinix Job Description.
- Parse salary để tạo `salary_expected_million_vnd`.
- Chỉ giữ salary rõ ràng cho clustering v1.
- Audit outlier, không xóa mặc định.
- Clean text/categorical cơ bản.
- Không remove salary mention mặc định.
- Split 90% train / 10% test.
- Xuất đầy đủ raw/clean train/test và audit files.

