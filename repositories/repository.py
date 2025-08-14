# repositories.py
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.models import Base, create_sequence_model

class SequenceRepository:
    def __init__(self, conn_str: str, table_name: str, df_schema_like: pd.DataFrame):
        self.engine = create_engine(conn_str, fast_executemany=True, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)
        # Tạo model theo schema thật của df_X_Y
        self.Model = create_sequence_model(table_name, df_schema_like)
        # Tạo bảng nếu chưa có
        Base.metadata.create_all(self.engine)

    def insert_dataframe(self, df: pd.DataFrame, chunksize: int = 50_000):
        """Ghi toàn bộ df_X_Y xuống SQL Server (rất nhanh, nhiều cột)."""
        # đảm bảo đúng thứ tự/đủ cột theo model
        cols = [c.name for c in self.Model.__table__.columns if c.name != "id"]
        data = df[cols]
        with self.engine.begin() as conn:
            data.to_sql(self.Model.__tablename__, conn, if_exists="append",
                        index=False, method="multi", chunksize=chunksize)