# Final Data Science - Salary Prediction Pipeline

Project hien tai duoc to chuc gon theo 3 file chinh:

- [notebooks/01_raw_data_eda.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/01_raw_data_eda.ipynb): get dataset + EDA tren raw data
- [notebooks/02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb): xu ly data end-to-end trong notebook
- [data_processing.py](/D:/H-Coding/Final-Data-Science/data_processing.py): ban script cua pipeline de chay bang command line neu can

Dataset su dung:
- `tinixai/vietnamese-job-descriptions`

Muc tieu cua phase nay:
- lay dataset ve
- phan tich raw data truoc khi clean
- parse salary va tao target `salary_expected_million_vnd`
- audit outlier
- clean text + remove salary leakage
- split train/test
- export 4 file du lieu dung format de nop mon

## Cau truc project

- `notebooks/`
  - `01_raw_data_eda.ipynb`
  - `02_data_processing.ipynb`
- `artifacts/`
  - `audit/`
  - `figures/`
  - `raw/`
  - `clean/`
- `data_processing.py`
- `README.md`

## Huong dan chay

### Cach 1: Chay bang notebook

#### Buoc 1. Raw EDA

Mo [01_raw_data_eda.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/01_raw_data_eda.ipynb) va chay tu tren xuong duoi.

Notebook nay se:
- download/load dataset
- kiem tra schema thuc te
- thong ke missing, duplicate, salary pattern
- do do dai text
- audit salary leakage trong text
- xuat bang audit va figure vao `artifacts/`

#### Buoc 2. Data processing

Mo [02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb) va chay tu tren xuong duoi.

Cell dau tien cua notebook se cai thu vien:

```python
%pip install pandas numpy pyarrow matplotlib seaborn datasets scikit-learn
```

Notebook co 1 cell config:

```python
config.sample_size = None
```

Quy uoc:
- `config.sample_size = None`: chay toan bo dataset
- `config.sample_size = 5000`: chay thu nhanh tren sample 5000 dong

Khuyen nghi:
- chay sample truoc de test pipeline
- sau khi on dinh thi doi lai `None` de chay full dataset

### Cach 2: Chay bang file Python

Neu muon chay bang script thay vi notebook:

Raw EDA:

```bash
python data_processing.py --mode eda --data-dir artifacts
```

Processing full:

```bash
python data_processing.py --mode process --data-dir artifacts
```

Processing voi sample:

```bash
python data_processing.py --mode process --data-dir artifacts --sample-size 5000
```

## Quy trinh xu ly du lieu

Pipeline trong [data_processing.py](/D:/H-Coding/Final-Data-Science/data_processing.py) va [02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb) giu dung thu tu sau:

1. Load dataset
- uu tien doc file cache local cua Hugging Face neu da co
- neu chua co cache thi goi `load_dataset("tinixai/vietnamese-job-descriptions")`

2. Parse salary va tao target
- chuan hoa chuoi `salary`
- nhan dien salary dang range, single, ambiguous, invalid
- quy doi USD/VND ve `million_vnd`
- tao:
  - `salary_raw_normalized`
  - `salary_pattern`
  - `salary_currency`
  - `salary_min`
  - `salary_max`
  - `salary_expected_million_vnd`
  - `salary_parse_status`

3. Loai target khong hop le
- chi giu cac dong:
  - parse thanh cong
  - target > 0
  - `salary_min <= salary_max`
  - target khong null, khong infinite

4. Audit outlier
- tinh:
  - Q1, Q3, IQR
  - P1, P99
- tao co:
  - `is_salary_outlier`
  - `outlier_reason`
- mac dinh:
  - chi gan co outlier
  - khong xoa outlier tu dong

5. Deduplicate
- drop duplicate theo `id`
- drop duplicate hoan toan
- gan co near-duplicate theo:
  - `job_title + company_name + location + salary`

6. Split train/test
- dung `train_test_split`
- co gang stratify theo bin cua `salary_expected_million_vnd` neu kha thi
- random seed mac dinh: `42`

7. Clean text va categorical
- lowercase
- normalize unicode
- bo HTML
- bo URL, email, phone
- bo ky tu rac va whitespace thua
- chuan hoa gia tri rong thanh `unknown`

8. Remove salary leakage
- ap dung cho:
  - `job_description`
  - `benefits`
  - `requirements`
- xoa cac pattern salary ro rang trong text
- giu lai cac thong tin benefit khong phai luong

9. Export file
- `artifacts/raw/raw_data_train.csv`
- `artifacts/raw/raw_data_test.csv`
- `artifacts/clean/clean_data_train.csv`
- `artifacts/clean/clean_data_test.csv`

## Output duoc tao

### 1. Data files

- [raw_data_train.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/raw_data_train.csv)
- [raw_data_test.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/raw_data_test.csv)
- [clean_data_train.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/clean_data_train.csv)
- [clean_data_test.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/clean_data_test.csv)

### 2. Audit files

Trong `artifacts/audit/` se co cac file quan trong nhu:
- `raw_schema_summary.csv`
- `raw_missing_summary.csv`
- `raw_text_length_summary.csv`
- `raw_salary_pattern_summary.csv`
- `raw_duplicate_summary.csv`
- `raw_top_values.csv`
- `raw_salary_leakage_summary.csv`
- `raw_salary_outlier_examples.csv`
- `raw_rule_candidates.json`
- `salary_outlier_summary.csv`
- `salary_outlier_examples.csv`
- `near_duplicate_audit.csv`
- `clean_salary_leakage_summary.csv`
- `train_test_target_distribution.csv`

### 3. Figures

Trong `artifacts/figures/` se co cac figure phuc vu bao cao:
- `raw_salary_patterns.png`
- `raw_text_length_boxplot.png`
- `salary_leakage_ratio.png`
- `salary_string_length_hist.png`
- `top_job_industry.png`
- `top_location.png`
- `top_job_type.png`
- `top_experience_level.png`
- `top_education_level.png`
- `top_job_position.png`
- `target_distribution_by_split.png`
- `salary_target_boxplot.png`
- `outlier_flag_ratio.png`

## Ghi chu quan trong

- Ty gia mac dinh: `25,000 VND/USD`
- Target chinh cua bai toan:
  - `salary_expected_million_vnd`
- Chi giu cac dong salary parse duoc ro rang de tao target
- Outlier duoc audit va gan co, khong bi xoa mac dinh
- Notebook `02_data_processing.ipynb` da ho tro chay full dataset
- Full run se rat nang va mat thoi gian; nen test bang sample truoc
