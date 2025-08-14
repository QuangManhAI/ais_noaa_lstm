from service_domain.data_processing import *
from repositories.repository import *

if __name__ == '__main__':
    file_path = '/home/quangmanh/Documents/thay_chien/2023_NOAA_AIS_logs_01.parquet'
    file_to = '/home/quangmanh/Documents/thay_chien/data_clean_3/'
    pf = read_data_parquet(file_path)
    all_mmsi = MMSI_unique(pf)

    
    processed_mmsi = get_processed_mmsi_ids(file_to)

    
    MMSIs = list(all_mmsi - processed_mmsi)

    func = partial(process_and_save, pf=pf, file_to=file_to)

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(tqdm(executor.map(func, MMSIs), total=len(MMSIs), desc="Processing__vessel__data!"))

    folder_path = '/home/quangmanh/Documents/thay_chien/data_clean_3/'
    all_dfs = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(folder_path, filename))
            all_dfs.append(df[['X', 'Y']])  # chỉ lấy cột cần thiết


    df_all = pd.concat(all_dfs, ignore_index=True)

    scaler = StandardScaler()
    scaler.fit(df_all[['X', 'Y']])

    import joblib
    joblib.dump(scaler, 'scaler_xy_from_1000_ships.pkl')


    folder_path = '/home/quangmanh/Documents/thay_chien/data_clean_3/'
    scaler_path = 'scaler_xy_from_1000_ships.pkl'

    df_train = to_data_train(folder_path, scaler_path)


    df = df_train

    feature_cols = [
    "SOG",
    "Heading_sin", "Heading_cos",
    "COG_sin", "COG_cos",
    "bearing_sin", "bearing_cos",
    "X_norm", "Y_norm",
    "hour_sin", "hour_cos",
    ]

    df_cleaned = build_sequence_samples_limited(
    df,
    feature_cols=feature_cols,
    seq_len=10,
    max_samples_per_group=100,
    max_total_groups=2
    )

    df_X_Y = pd.concat(df_cleaned, ignore_index=True)
    display(df_X_Y.head(3))

    conn_str = "mssql+pyodbc://sa:StrongPassword123!@localhost:1433/DB02?driver=ODBC+Driver+17+for+SQL+Server"
    repo = SequenceRepository(conn_str, table_name="ais_sequences", df_schema_like=df_X_Y)
    repo.insert_dataframe(df_X_Y)