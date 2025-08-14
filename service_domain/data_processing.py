import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from geopy.distance import geodesic
from pyproj import Transformer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from functools import partial
import os
from IPython.display import display
import joblib


def read_data_parquet(file_path):
    return pq.ParquetFile(file_path)

def write_data_parquet(df, filepath):
    df.to_parquet(filepath, index=False)

def MMSI_unique(pf):
    ship_set = set()
    for i in range(pf.num_row_groups):
        df_i = pf.read_row_group(i, columns=['MMSI']).to_pandas()
        ship_set.update(df_i['MMSI'].unique())
    return ship_set

def get_processed_mmsi_ids(folder_path):
    processed_files = [f for f in os.listdir(folder_path) if f.startswith("mmsi_") and f.endswith(".parquet")]
    return {int(f.split('_')[1].split('.')[0]) for f in processed_files}

def calc_trip_summary(df, speed_threshold_knots=2):
    df = df.sort_values('BaseDateTime').copy()
    df['time_diff'] = df['BaseDateTime'].diff().dt.total_seconds()

    df['LAT_prev'] = df['LAT'].shift()
    df['LON_prev'] = df['LON'].shift()

    df['distance_km'] = df.apply(
        lambda row: geodesic((row['LAT_prev'], row['LON_prev']), (row['LAT'], row['LON'])).km
        if pd.notnull(row['LAT_prev']) and pd.notnull(row['LON_prev']) else 0,
        axis=1
    )

    total_time_hr = df['time_diff'].sum() / 3600
    total_distance_km = df['distance_km'].sum()
    sog_avg_knot = df['SOG'].mean()
    sog_kmh = sog_avg_knot * 1.852
    real_avg_speed = total_distance_km / total_time_hr if total_time_hr > 0 else 0

    # Kiểm tra tốc độ trung bình
    is_valid_speed = sog_avg_knot > speed_threshold_knots

    summary_row = pd.DataFrame([{
        'start_time': df['BaseDateTime'].iloc[0],
        'end_time': df['BaseDateTime'].iloc[-1],
        'duration_hr': total_time_hr,
        'distance_km': total_distance_km,
        'real_avg_speed_kmh': real_avg_speed,
        'sog_avg_knot': sog_avg_knot,
        'sog_kmh': sog_kmh,
        'Physical': np.isclose(real_avg_speed, sog_kmh, atol=5),
        'Valid_Speed': is_valid_speed
    }])

    df = pd.concat([df, summary_row], ignore_index=True)
    return df


transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def LAT_LON_process(df, scaler=None):
    df['X'], df['Y'] = transformer.transform(df['LON'].values, df['LAT'].values)

    if scaler is None:
        scaler = MinMaxScaler()
        df[['X_norm', 'Y_norm']] = scaler.fit_transform(df[['X', 'Y']])
    else:
        df[['X_norm', 'Y_norm']] = scaler.transform(df[['X', 'Y']])
    
    df['deltaX'] = df['X'].diff()
    df['deltaY'] = df['Y'].diff()

    df['bearing'] = (np.degrees(np.arctan2(df['deltaY'], df['deltaX'])) + 360) % 360
    df['bearing_sin'] = np.sin(np.radians(df['bearing']))
    df['bearing_cos'] = np.cos(np.radians(df['bearing']))

    df.dropna(subset=['deltaX', 'deltaY'], inplace=True)


def angular_diff(a, b):
    d = (a - b + 180) % 360 - 180
    return d 

def cog_heading_process(df):

    df['COG_rad'] = np.radians(df['COG'])
    df['COG_sin'] = np.sin(df['COG_rad'])
    df['COG_cos'] = np.cos(df['COG_rad'])

    df['Heading_rad'] = np.radians(df['Heading'])
    df['Heading_sin'] = np.sin(df['Heading_rad'])
    df['Heading_cos'] = np.cos(df['Heading_rad'])

    df['COG_prev'] = df['COG'].shift()
    df['Heading_prev'] = df['Heading'].shift()

    df['delta_COG'] = df.apply(lambda row: angular_diff(row['COG'], row['COG_prev']) if pd.notnull(row['COG_prev']) else 0, axis = 1)
    df['delta_Heading'] = df.apply(lambda row: angular_diff(row['Heading'], row['Heading_prev']) if pd.notnull(row['Heading_prev']) else 0, axis=1)

    if 'bearing' in df.columns:
        df['bearing_diff'] = df.apply(lambda row: angular_diff(row['COG'], row['bearing']), axis = 1)
    df.drop(columns=['COG_prev', 'Heading_prev'], inplace=True)

def time_process(df):

    df['time_diff'] = df['BaseDateTime'].diff().dt.total_seconds().fillna(0)

    df['hour'] = df['BaseDateTime'].dt.hour + df['BaseDateTime'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)


def extract_and_save_mmsi(pf, mmsi):
    vessel_data = []
    for group in range(pf.num_row_groups):
        df_i = pf.read_row_group(group).to_pandas()
        filtered = df_i[df_i['MMSI'] == mmsi]
        if not filtered.empty:
            vessel_data.append(filtered)

    if not vessel_data:
        return None
    full_df = pd.concat(vessel_data, ignore_index=True)
    
    summary = calc_trip_summary(full_df)
    if not summary.iloc[-1]['Physical'] or not summary.iloc[-1]['Valid_Speed']:  
        return None

    full_df = full_df.iloc[:-1]  

    LAT_LON_process(full_df)
    cog_heading_process(full_df)
    time_process(full_df)
    if len(full_df) < 15:
        return None
    return full_df

valid_count = 0
max_vessels = 10 # số tàu hợp lệ muốn chọnS

def process_and_save(mmsi, pf, file_to):
    global valid_count 
    if valid_count >= max_vessels:
        return 

    try:
        df_clean = extract_and_save_mmsi(pf, mmsi)
        if df_clean is not None:
            file_out = f"{file_to}mmsi_{mmsi}.parquet"
            write_data_parquet(df_clean, file_out)
            print(f"MMSI {mmsi} -> Done")
            valid_count += 1 
        else:
            print(f"MMSI {mmsi} -> Skipped (Not Physical or Invalid Speed)")
            with open("MMSI_info.txt", "a") as f:
                f.write(f"{mmsi}\n")
    except Exception as e:
        print(f"Error processing MMSI {mmsi}: {e}")
        with open("error_log.txt", "a") as f:
            f.write(f"MMSI {mmsi}: {str(e)}\n")
        with open("MMSI_info.txt", "a") as f:
            f.write(f"{mmsi}\n")

            


def to_data_train(folder_path, scaler_path):
    all_dataframes = []

    scaler = joblib.load(scaler_path)

    for filename in tqdm(os.listdir(folder_path)):
        if not filename.endswith(".parquet"):
            continue

        full_path = os.path.join(folder_path, filename)

        try:
            table = pq.read_table(full_path)
            df = table.to_pandas()

            df = df[(df['Heading'] >= 0) & (df['Heading'] <= 360)]
            df = df[(df['COG'] >= 0) & (df['COG'] <= 360)]
            df = df.dropna(subset=['LAT', 'LON'])

            if 'X' in df.columns and 'Y' in df.columns:
                df[['X_norm', 'Y_norm']] = scaler.transform(df[['X', 'Y']])

            if len(df) >= 500:
                all_dataframes.append(df)

        except Exception as e:
            pass
            # print(f"Lỗi khi xử lý file {filename}: {e}")

    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        print(f" Số tàu hợp lệ: {len(all_dataframes)} / {len(os.listdir(folder_path))}")
        print(f" Tổng số dòng dữ liệu: {len(merged_df):,}")
        return merged_df
    else:
        print(" Không có tàu nào đủ điều kiện!")
        return pd.DataFrame()



def build_sequence_samples_limited(
    df: pd.DataFrame,
    feature_cols: list,
    seq_len: int = 10,
    stop_speed: float = 6,
    max_time_gap: float = 360,
    max_samples_per_group: int = 500_000,
    max_total_groups: int = 3
) -> list[pd.DataFrame]:
    """
    Cắt dữ liệu AIS thành các đoạn liên tục để huấn luyện mô hình dự đoán tọa độ (X_norm, Y_norm).
    Trả về DataFrame chứa X_norm, Y_norm. (Không bao gồm X, Y thực tế).
    """
    df = df.sort_values(['MMSI', 'BaseDateTime']).copy()
    groups = []
    current_group = []
    current_count = 0

    for mmsi, group in df.groupby('MMSI'):
        group = group[group['SOG'] > stop_speed].reset_index(drop=True)
        if len(group) < seq_len + 1:
            continue

        group['time_diff'] = group['BaseDateTime'].diff().dt.total_seconds().fillna(0)

        for i in range(len(group) - seq_len):
            window = group.iloc[i:i + seq_len + 1]
            if window['time_diff'].iloc[1:seq_len].max() > max_time_gap:
                continue

            features_seq = window[feature_cols].iloc[:seq_len].values.flatten()
            target_norm = window[['X_norm', 'Y_norm']].iloc[seq_len].values


            # mới sửa dòng này để hết bug không biết chạy có ra không.
            row = np.concatenate([np.asarray(features_seq), np.asarray(target_norm)])


            current_group.append(row)
            current_count += 1

            if current_count >= max_samples_per_group:
                feature_names = [f'{col}_t{t}' for t in range(seq_len) for col in feature_cols]
                df_group = pd.DataFrame(current_group, columns=feature_names + ['X_norm', 'Y_norm'])
                groups.append(df_group)

                current_group = []
                current_count = 0

                if len(groups) >= max_total_groups:
                    return groups

    if current_group:
        feature_names = [f'{col}_t{t}' for t in range(seq_len) for col in feature_cols]
        df_group = pd.DataFrame(current_group, columns=feature_names + ['X_norm', 'Y_norm'])
        groups.append(df_group)

    return groups



# if __name__ == '__main__':
#     file_path = '/home/quangmanh/Documents/thay_chien/2023_NOAA_AIS_logs_01.parquet'
#     file_to = '/home/quangmanh/Documents/thay_chien/data_clean_2/'
#     pf = read_data_parquet(file_path)
#     all_mmsi = MMSI_unique(pf)

    
#     processed_mmsi = get_processed_mmsi_ids(file_to)

    
#     MMSIs = list(all_mmsi - processed_mmsi)

#     func = partial(process_and_save, pf=pf, file_to=file_to)

#     with ThreadPoolExecutor(max_workers=12) as executor:
#         list(tqdm(executor.map(func, MMSIs), total=len(MMSIs), desc="Processing__vessel__data!"))

#     folder_path = '/home/quangmanh/Documents/thay_chien/data_clean_2/'
#     all_dfs = []

#     for filename in tqdm(os.listdir(folder_path)):
#         if filename.endswith(".parquet"):
#             df = pd.read_parquet(os.path.join(folder_path, filename))
#             all_dfs.append(df[['X', 'Y']])  # chỉ lấy cột cần thiết


#     df_all = pd.concat(all_dfs, ignore_index=True)

#     scaler = StandardScaler()
#     scaler.fit(df_all[['X', 'Y']])

#     import joblib
#     joblib.dump(scaler, 'scaler_xy_from_1000_ships.pkl')


#     folder_path = '/home/quangmanh/Documents/thay_chien/data_clean_2/'
#     scaler_path = 'scaler_xy_from_1000_ships.pkl'

#     df_train = to_data_train(folder_path, scaler_path)


#     df = df_train

#     feature_cols = [
#     "SOG",
#     "Heading_sin", "Heading_cos",
#     "COG_sin", "COG_cos",
#     "bearing_sin", "bearing_cos",
#     "X_norm", "Y_norm",
#     "hour_sin", "hour_cos",
#     ]

#     df_cleaned = build_sequence_samples_limited(
#     df,
#     feature_cols=feature_cols,
#     seq_len=10,
#     max_samples_per_group=270_000,
#     max_total_groups=2
#     )

#     df_X_Y = pd.concat(df_cleaned, ignore_index=True)
#     display(df_X_Y.head(3))

