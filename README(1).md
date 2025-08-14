# AIS 2023 Processing & SQL Server Storage

## Mô tả
Dự án xử lý dữ liệu AIS (Automatic Identification System) năm 2023, trích xuất đặc trưng từ dữ liệu thô và lưu trữ kết quả xuống **SQL Server**.

Chức năng chính:
- Đọc dữ liệu AIS định dạng **Parquet**.
- Tiền xử lý và tính toán các đặc trưng t0 → t9 (vị trí, tốc độ, hướng, góc…).
- Sinh dữ liệu chuỗi mẫu (`df_X_Y`) cho huấn luyện mô hình dự đoán vị trí.
- Lưu kết quả xuống SQL Server với mô hình 3 layers:
  - **models/**: định nghĩa class ORM ánh xạ bảng DB.
  - **repositories/**: thao tác CRUD với DB.
  - **main.py**: điều phối luồng xử lý.

---

## Cấu trúc thư mục
```
project/
├─ main.py                    # Điểm vào chương trình
├─ data_processing.py         # Xử lý AIS, tính toán đặc trưng
├─ models/
│   └─ models.py              # Định nghĩa ORM model động từ DataFrame
├─ repositories/
│   └─ repository.py          # Lớp SequenceRepository ghi xuống SQL Server
└─ README.md                  # Tài liệu dự án
```

---

## Yêu cầu môi trường
- Python 3.10+
- Thư viện:
  ```bash
  pip install -r requirements.txt
  ```
- **SQL Server** cài sẵn driver ODBC 18:
  - Windows: [ODBC Driver 18 for SQL Server](https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server)
  - Linux: cài qua `apt` hoặc `yum`.

---

## Cách chạy
1. **Chuẩn bị dữ liệu AIS**
   - Đặt file Parquet AIS 2023 (hoặc tháng/ngày cần) vào thư mục dữ liệu.
   - Hoặc tải từ Kaggle:
     ```bash
     kaggle datasets download -d <dataset-id> -f <filename>
     unzip <filename>
     ```

2. **Cấu hình kết nối SQL Server**
   - Sửa `conn_str` trong `main.py`:
     ```python
     conn_str = "mssql+pyodbc://localhost\\SQLEXPRESS/AISDB?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
     ```

3. **Chạy chương trình**
   ```bash
   python main.py
   ```
   Chương trình sẽ:
   - Đọc dữ liệu AIS thô.
   - Tiền xử lý, tính toán đặc trưng.
   - Sinh `df_X_Y` (chuỗi mẫu huấn luyện).
   - Ghi dữ liệu xuống SQL Server.

---

## Luồng xử lý
1. `main.py` gọi hàm từ `data_processing.py` để:
   - Đọc file AIS.
   - Lọc tàu hợp lệ.
   - Tính toán đặc trưng (LAT/LON → X/Y, SOG, COG, Heading, Bearing…).
   - Sinh dữ liệu chuỗi mẫu.
2. `repositories/SequenceRepository` nhận DataFrame và ghi xuống DB theo lô, tự tính `chunksize` để tránh giới hạn **2100 parameters** của SQL Server.
3. Bảng lưu trữ tự tạo schema dựa trên DataFrame.

---

## Lưu ý
- Dữ liệu AIS rất lớn (nhiều GB), cần **chunk xử lý** và **ghi theo lô** để tránh hết RAM hoặc bị **OOM killed**.
- SQL Server giới hạn ~2100 tham số mỗi câu lệnh → đã xử lý trong repository.
- Nếu chỉ cần một phần dữ liệu từ Kaggle, nên tải theo file/tháng để tiết kiệm dung lượng.

---

## Làm việc nhóm
- Sử dụng Git để quản lý code.
- Chia module theo 3 layers:
  - **models**: định nghĩa class bảng DB.
  - **repositories**: truy xuất dữ liệu.
  - **service / main**: xử lý và điều phối.
- Mỗi thành viên có thể phụ trách 1 layer hoặc một nhóm chức năng.

---
