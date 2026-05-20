# Data Processing Verification Guide

Tai lieu nay mo ta chi tiet toan bo quy trinh xu ly du lieu cho de tai:

- Du doan muc luong ky vong tu Vietnamese Job Descriptions

File nay co 3 muc dich:
- mo ta day du quy trinh data processing va target creation
- huong dan chay lai pipeline bang notebook hoac script
- ghi ro ket qua verify hien tai de biet phan nao da dung, phan nao can rerun

## 1. File lien quan

File chinh trong project:
- [notebooks/01_raw_data_eda.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/01_raw_data_eda.ipynb): notebook lay dataset va EDA tren raw data
- [notebooks/02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb): notebook chay toan bo quy trinh processing
- [data_processing.py](/D:/H-Coding/Final-Data-Science/data_processing.py): ban script cua pipeline

Thu muc output:
- [artifacts](/D:/H-Coding/Final-Data-Science/artifacts)
- [artifacts/audit](/D:/H-Coding/Final-Data-Science/artifacts/audit)
- [artifacts/figures](/D:/H-Coding/Final-Data-Science/artifacts/figures)
- [artifacts/raw](/D:/H-Coding/Final-Data-Science/artifacts/raw)
- [artifacts/clean](/D:/H-Coding/Final-Data-Science/artifacts/clean)

## 2. Muc tieu cua processing phase

Quy trinh nay phai dat duoc 4 dau ra chinh:
- `raw_data_train.csv`
- `raw_data_test.csv`
- `clean_data_train.csv`
- `clean_data_test.csv`

Dong thoi phai:
- kiem tra chat luong raw data truoc khi clean
- parse duoc `salary` thanh target so hoc
- loai bo dong khong tao duoc target hop le
- danh dau outlier nhung khong xoa mac dinh
- xoa salary leakage khoi text modeling
- luu lai audit files va figures cho bao cao

## 3. Nguon du lieu va cach load

Dataset nguon:
- `tinixai/vietnamese-job-descriptions`

Pipeline uu tien load theo thu tu:
1. Neu dataset da duoc cache local trong Hugging Face cache, doc truc tiep file `data.parquet`
2. Neu chua co cache, goi:

```python
from datasets import load_dataset
ds = load_dataset("tinixai/vietnamese-job-descriptions")
```

Ly do:
- cach nay nhanh hon khi chay lai nhieu lan
- tranh phu thuoc vao network sau khi dataset da tai ve

## 4. Quy trinh xu ly du lieu chi tiet

### Buoc 1. Raw EDA

Raw EDA duoc chay truoc cleaning.

Muc dich:
- xac nhan schema thuc te
- thong ke missing
- thong ke duplicate
- thong ke pattern cua `salary`
- do muc leakage trong text
- tao bang audit va figure de dua vao bao cao

Raw EDA tao ra:
- `raw_schema_summary.csv`
- `raw_missing_summary.csv`
- `raw_text_length_summary.csv`
- `raw_salary_pattern_summary.csv`
- `raw_duplicate_summary.csv`
- `raw_top_values.csv`
- `raw_salary_leakage_summary.csv`
- `raw_salary_outlier_examples.csv`
- `raw_rule_candidates.json`

### Buoc 2. Chuan hoa chuoi salary

Cac xu ly chinh:
- lowercase
- normalize unicode
- bo khoang trang thua
- chuan hoa dau phan cach nhu `-`, `~`, `to`
- chuan hoa token don vi nhu:
  - `usd`, `$`
  - `trieu`, `triệu`, `tr`
  - `vnd`, `vnđ`

Vi du:
- `26 - 36 triệu` -> `26 - 36 trieu`
- `1,200 usd` -> `1200 usd`

### Buoc 3. Phan loai salary

Moi dong `salary` duoc phan loai thanh:
- `range_vnd`
- `single_vnd`
- `range_usd`
- `single_usd`
- `range_unknown`
- `single_unknown`
- `ambiguous`
- `invalid`
- `missing`

Y nghia:
- `range_*`: co khoang luong min-max
- `single_*`: co 1 muc luong don
- `unknown`: co so nhung khong xac dinh duoc don vi
- `ambiguous`: dang mo ho nhu `thoa thuan`, `cạnh tranh`
- `invalid`: khong parse duoc co nghia

### Buoc 4. Tao target

Sau khi parse salary, pipeline tao cac cot:
- `salary_raw_normalized`
- `salary_pattern`
- `salary_currency`
- `salary_min`
- `salary_max`
- `salary_expected_million_vnd`
- `salary_parse_status`

Cong thuc target:

```text
salary_expected_million_vnd = (salary_min + salary_max) / 2
```

Neu la salary don:
- `salary_min = salary_max = muc luong do`

Quy doi don vi:
- VND -> chia `1_000_000`
- USD -> nhan ty gia mac dinh `25,000 VND/USD`, sau do doi sang `million_vnd`

Vi du:
- `4000 usd` -> `100.0` trieu VND
- `26 - 36 triệu` -> `31.0`

### Buoc 5. Loai dong khong tao duoc target hop le

Chi giu lai cac dong thoa man:
- `salary_parse_status == "valid"`
- `salary_expected_million_vnd > 0`
- `salary_min <= salary_max`
- target khong null
- target khong infinite

He qua:
- nhom chon huong conservative
- uu tien target sach hon thay vi co gang cuu moi dong salary mo ho

### Buoc 6. Outlier audit

Outlier duoc xu ly theo huong:
- audit va gan co
- khong xoa mac dinh

Thong ke duoc tinh:
- `Q1`
- `Q3`
- `IQR`
- `P1`
- `P99`

Cot tao them:
- `is_salary_outlier`
- `outlier_reason`

Rule:
- nho hon nguong IQR lower hoac percentile P1
- lon hon nguong IQR upper hoac percentile P99

Y nghia:
- nhom van giu du lieu cuc tri de danh gia sau
- modeling phase co the thu 2 kich ban: giu outlier va loai outlier

### Buoc 7. Deduplicate

Pipeline xu ly 3 muc:
- duplicate theo `id`
- duplicate hoan toan theo hang
- near duplicate theo:
  - `job_title + company_name + location + salary`

Rule hien tai:
- drop duplicate theo `id`
- drop duplicate hoan toan
- near duplicate chi gan co trong audit file, khong xoa hang loat mac dinh

### Buoc 8. Split train/test

Split su dung:
- `train_test_split`
- `test_size = 0.2`
- `random_state = 42`

Neu phan phoi target cho phep:
- tao bin theo `salary_expected_million_vnd`
- stratify theo bin de train/test can bang hon

### Buoc 9. Clean text

Ap dung cho cac cot:
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

Rule:
- lowercase
- normalize unicode
- bo HTML
- bo URL
- bo email
- bo phone
- bo ky tu lap vo nghia
- chuan hoa whitespace
- chuyen rong thanh `unknown`

Khong lam o phase nay:
- stemming
- stopword removal
- tokenization
- TF-IDF
- encoding feature

### Buoc 10. Remove salary leakage

Chi ap dung cho:
- `job_description`
- `benefits`
- `requirements`

Muc tieu:
- tranh cho mo hinh hoc truc tiep tu thong tin salary xuat hien trong text

Xoa:
- so tien ro rang
- khoang luong
- cum salary co gan gia tri luong

Vi du:
- `15 triệu`
- `20.000.000`
- `1200 usd`
- `23-28 triệu`

Sau step nay:
- clean data phai khong con salary leakage ro rang trong 3 cot text chinh

### Buoc 11. Export

4 file du lieu:
- [raw_data_train.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/raw_data_train.csv)
- [raw_data_test.csv](/D:/H-Coding/Final-Data-Science/artifacts/raw/raw_data_test.csv)
- [clean_data_train.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/clean_data_train.csv)
- [clean_data_test.csv](/D:/H-Coding/Final-Data-Science/artifacts/clean/clean_data_test.csv)

Y nghia:
- `raw_*`: da co target hop le va split, nhung chua clean text sau cung
- `clean_*`: da clean text va xoa leakage, san sang cho feature engineering

## 5. Cach chay lai pipeline

### Bang notebook

1. Mo [01_raw_data_eda.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/01_raw_data_eda.ipynb)
2. Chay raw EDA tu tren xuong duoi
3. Mo [02_data_processing.ipynb](/D:/H-Coding/Final-Data-Science/notebooks/02_data_processing.ipynb)
4. Cell dau tien cai thu vien
5. Chinh:

```python
config.sample_size = 5000
```

de test nhanh, hoac:

```python
config.sample_size = None
```

de chay full dataset

### Bang script

Raw EDA:

```bash
python data_processing.py --mode eda --data-dir artifacts
```

Processing sample:

```bash
python data_processing.py --mode process --data-dir artifacts --sample-size 5000
```

Processing full:

```bash
python data_processing.py --mode process --data-dir artifacts
```

## 6. Ket qua verify hien tai

### Da verify duoc

1. File output ton tai va doc duoc
- `raw_data_train.csv` ton tai
- `raw_data_test.csv` ton tai
- `clean_data_train.csv` ton tai
- `clean_data_test.csv` ton tai

2. Tren preview 5000 dong cua cac file export:
- tat ca `salary_parse_status` deu la `valid`
- tat ca target deu duong
- tat ca `salary_min <= salary_max`

3. Leakage removal dang hoat dong dung tren clean data preview
- preview `raw_data_*`: leakage ratio tren 3 cot text chinh con khoang `0.17`
- preview `clean_data_*`: leakage ratio tren 3 cot text chinh = `0.0`

4. Tren sample 20,000 dong parse truc tiep tu dataset goc:
- `valid`: `12,522`
- `invalid`: `6,945`
- `unknown_currency`: `492`
- `ambiguous`: `41`

Mot so salary parse mau:
- `26 - 36 triệu` -> `31.0`
- `15 - 35 triệu` -> `25.0`
- `4000 usd` -> `100.0`
- `1200 usd` -> `30.0`

5. Tren sample 50,000 dong, split hien tai can bang tot
- mean train: `13.9366`
- mean test: `13.8554`
- median train = median test = `12.5`

Ket luan tu cac buoc tren:
- logic parser va target creation hien tai la hop ly theo huong conservative
- leakage removal dang dung
- split logic trong code hien tai co ve dung khi test lai tren sample lon

### Van de phat hien khi verify full artifact hien tai

File audit full trong `artifacts/` dang cho:
- train mean: `30.7149`
- test mean: `13.6802`

Do lech nay qua lon va khong phu hop voi ket qua verify tren sample 50k.

Dieu nay cho thay:
- code hien tai co kha nang da dung
- nhung full artifact dang luu trong `artifacts/` co the la output cu/stale tu mot lan run truoc, hoac mot lan run dang do

Noi cach khac:
- **pipeline logic hien tai: tam on**
- **full artifact hien tai trong `artifacts/`: chua nen coi la da verify xong 100%**

## 7. Danh gia tong the: processing va target creation da dung chua?

Danh gia trung thuc:

- **Dung o muc logic xu ly**:
  - parser salary hoat dong hop ly voi pattern pho bien
  - target `salary_expected_million_vnd` duoc tao dung theo yeu cau bai
  - clean text va leakage removal dung theo muc tieu do an
  - outlier duoc audit thay vi xoa vo co

- **Chua the ket luan full output hien tai da dung hoan toan**:
  - vi train/test distribution trong artifact full dang lech bat thuong
  - can rerun full processing va kiem tra lai file `train_test_target_distribution.csv`

## 8. Viec nen lam ngay

De chot dataset clean dung va an toan cho modeling:

1. Chay lai full processing bang code hien tai
2. Kiem tra lai:
   - `train_test_target_distribution.csv`
   - mean va median train/test
   - leakage ratio cua clean files
3. Neu train/test mean sau rerun gan nhau hon, co the coi full pipeline da on

## 9. Ket luan ngan

Hien tai:
- quy trinh xu ly data duoc thiet ke dung huong
- target creation theo yeu cau mon hoc la dung
- clean/leakage removal dang hoat dong dung

Nhung:
- full artifact dang luu trong `artifacts/` can duoc rerun va verify lai 1 lan cuoi truoc khi dua sang phase modeling
