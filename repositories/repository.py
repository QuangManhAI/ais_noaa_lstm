# repositories/repository.py
from __future__ import annotations
import pandas as pd
from typing import Iterable, List, Sequence
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models.models import Base, create_sequence_model


class SequenceRepository:
    """
    Repository cho bảng 'sequence' (rộng, nhiều cột) sinh từ df_X_Y.
    - Tạo model động theo schema của DataFrame (không cần liệt kê 100+ cột).
    - Insert theo lô, tự tránh giới hạn 2100 tham số của SQL Server.
    - Upsert bằng bảng staging + MERGE theo key_cols (vd: ['sample_id'] hoặc tuỳ bạn).
    """
    def __init__(self, conn_str: str, table_name: str, df_schema_like: pd.DataFrame):
        self.engine = create_engine(conn_str, fast_executemany=True, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)

        # Model động theo đúng schema df_X_Y
        self.Model = create_sequence_model(table_name, df_schema_like)

        # Tạo bảng nếu chưa có
        Base.metadata.create_all(self.engine)

    # -------------------- helpers --------------------
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bổ sung cột thiếu cho khớp schema model, sắp xếp đúng thứ tự."""
        df = df.copy()
        cols_model: List[str] = [c.name for c in self.Model.__table__.columns if c.name != "id"]
        for c in cols_model:
            if c not in df.columns:
                df[c] = None
        return df[cols_model]

    def _safe_chunksize(self, ncols: int, req: int | None) -> int:
        """
        SQL Server: tối đa ~2100 parameters/statement.
        Với insert theo lô, params ≈ rows * ncols.
        Chọn chunksize an toàn nếu chưa chỉ định.
        """
        if req is not None and req > 0:
            return req
        # dùng 2000 thay vì 2100 để có đệm
        return max(1, 2000 // max(1, ncols))

    # -------------------- public APIs --------------------
    def insert_dataframe(self, df: pd.DataFrame, chunksize: int | None = None, cast_float32: bool = True) -> int:
        """
        Ghi DataFrame xuống SQL Server (append).
        - Không dùng method="multi" để tránh vượt 2100 tham số.
        - Chọn chunksize an toàn theo số cột.
        """
        data = self._ensure_columns(df)
        if cast_float32:
            # giảm tiêu thụ RAM/IO
            for c in data.columns:
                if pd.api.types.is_float_dtype(data[c]):
                    data[c] = data[c].astype("float32")

        safe_chunk = self._safe_chunksize(len(data.columns), chunksize)

        with self.engine.begin() as conn:
            data.to_sql(
                self.Model.__tablename__,
                conn,
                if_exists="append",
                index=False,
                chunksize=safe_chunk,
                method=None,           # QUAN TRỌNG: tránh 'multi' với SQL Server
            )
        return len(data)

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        key_cols: Sequence[str],
        staging_table: str = "seq_stage",
        chunksize: int | None = None,
        cast_float32: bool = True,
        drop_stage: bool = True,
    ) -> int:
        """
        Upsert bằng MERGE theo các cột khoá 'key_cols'.
        1) Đổ df vào bảng staging
        2) MERGE vào bảng chính
        """
        if not key_cols:
            raise ValueError("key_cols không được rỗng (ví dụ: ['MMSI','BaseDateTime'] hoặc ['sample_id']).")

        data = self._ensure_columns(df)
        if cast_float32:
            for c in data.columns:
                if pd.api.types.is_float_dtype(data[c]):
                    data[c] = data[c].astype("float32")

        safe_chunk = self._safe_chunksize(len(data.columns), chunksize)

        # 1) Ghi staging
        with self.engine.begin() as conn:
            data.to_sql(
                staging_table, conn, if_exists="replace", index=False,
                chunksize=safe_chunk, method=None
            )

            # 2) MERGE
            tab = self.Model.__tablename__
            cols = list(data.columns)  # tất cả cột (không có id)
            on_clause = " AND ".join([f"tgt.{k}=src.{k}" for k in key_cols])
            set_clause = ", ".join([f"tgt.{c}=src.{c}" for c in cols if c not in key_cols])
            insert_cols = ", ".join(cols)
            insert_vals = ", ".join([f"src.{c}" for c in cols])

            merge_sql = f"""
            MERGE dbo.{tab} AS tgt
            USING dbo.{staging_table} AS src
              ON {on_clause}
            WHEN MATCHED THEN UPDATE SET {set_clause}
            WHEN NOT MATCHED BY TARGET THEN
              INSERT ({insert_cols})
              VALUES ({insert_vals});
            """
            conn.execute(text(merge_sql))

            # 3) Dọn staging (tùy chọn)
            if drop_stage:
                conn.execute(text(f"DROP TABLE dbo.{staging_table};"))

        return len(data)