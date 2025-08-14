# models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Float

Base = declarative_base()

def create_sequence_model(table_name: str, df) -> type:
    """
    Tạo ORM class động khớp mọi cột của df_X_Y (rất nhiều feature_tk + X_norm/Y_norm).
    Dùng Float cho đơn giản/nhanh; nếu cần có thể map kiểu cụ thể hơn sau.
    """
    attrs = {
        "__tablename__": table_name,
        "id": Column(Integer, primary_key=True, autoincrement=True),
    }
    for col in df.columns:
        # bảo đảm tên cột hợp lệ với SQL Server (nếu có ký tự lạ thì tự xử thêm)
        attrs[col] = Column(Float)
    Model = type("SequenceSample", (Base,), attrs)
    return Model