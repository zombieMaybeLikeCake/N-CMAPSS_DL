import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
def condition_scaler(df_train, sensor_names):
    """ 根據操作條件標準化數據 """
    scaler = StandardScaler()
    scaler.fit(df_train[sensor_names])
    df_train[sensor_names] = scaler.transform(df_train[sensor_names])
    # df_test[sensor_names] = scaler.transform(df_test[sensor_names])
    return df_train

def exponential_smoothing(df, sensors, alpha=0.4):
    """ 指數平滑處理 """
    df = df.copy()
    df[sensors] = df[sensors].ewm(alpha=alpha).mean()
    return df
def create_time_windows_classification(df, feature_columns, sequence_length, target_column="class"):
    """
    建立時間序列的滑動視窗 (time windows)，輸出 y 為 `class` 分類標籤。

    :param df: DataFrame, 包含所有數據
    :param feature_columns: 需要作為 X 特徵的欄位
    :param sequence_length: 時間序列長度 (32)
    :param target_column: 預測目標欄位，預設為 'class'
    :return: X (輸入特徵), Y (分類標籤)
    """
    X, Y = [], []

    # 確保索引是連續數字，避免 loc[] 出錯
    df = df.reset_index(drop=True)

    for k in range(len(df) - sequence_length):
        if k + sequence_length >= len(df):  # 避免索引超界
            break

        X_seq = df.iloc[k:k+sequence_length][feature_columns].values  # 改用 iloc
        Y_label = df.iloc[k+sequence_length][target_column]  # 改用 iloc

        # 檢查是否有 NaN 值
        if np.isnan(X_seq).any() or pd.isnull(Y_label):
            continue  # 跳過有缺失值的資料

        X.append(X_seq.astype(np.float32))  # 轉為 float32
        Y.append(int(Y_label))  # 轉為 int

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)  # 確保數據型態正確

def get_data_for_classification_with_segments(df, sequence_length=32, test_size=0.1):
    """
    根據 `segment` 和 `segment_day` 分割資料集，建立符合分類問題的時間窗口數據。

    :param df: 包含 `segment`, `segment_day`, `class` 的 DataFrame
    :param sequence_length: 時間序列長度 (預設 32 天)
    :param test_size: 測試集比例
    :return: x_train, y_train, x_test, y_test, num_classes
    """
    if "segment" not in df.columns or "segment_day" not in df.columns or "class" not in df.columns:
        raise ValueError("資料集必須包含 `segment`, `segment_day`, `class` 欄位")

    # 先過濾 NaN 資料
    df = df.dropna().reset_index(drop=True)

    # 按 `segment` 和 `segment_day` 排序
    df = df.sort_values(by=["segment", "segment_day"]).reset_index(drop=True)

    # 取得數值型欄位 (排除 `segment`, `segment_day`, `class`)
    feature_columns = [col for col in df.columns if col not in ["segment", "segment_day", "class"]]
    df = condition_scaler(df, feature_columns)
    df = exponential_smoothing(df, feature_columns, alpha=0.1)
    if len(feature_columns) == 0:
        raise ValueError("沒有可用的數值特徵，請檢查 `seatempclasssegment.csv`")

    # 初始化 X, Y
    X_all, Y_all = [], []

    # 依 `segment` 建立獨立的時間序列
    for segment_id, segment_df in df.groupby("segment"):
        segment_df = segment_df.reset_index(drop=True)

        # 確保 segment 內部資料足夠長
        if len(segment_df) < sequence_length:
            continue

        for k in range(len(segment_df) - sequence_length):
            if k + sequence_length >= len(segment_df):
                break

            X_seq = segment_df.iloc[k:k+sequence_length][feature_columns].values.astype(np.float32)
            Y_label = segment_df.iloc[k+sequence_length]["class"]

            try:
                Y_label = int(Y_label)
            except ValueError:
                print(f"警告：`class` 欄位出現非整數數據 (segment {segment_id})，跳過")
                continue

            X_all.append(X_seq)
            Y_all.append(Y_label)

    # 轉換為 NumPy 陣列
    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.int64)

    # **確保 `class` 在合法範圍內**
    num_classes = np.max(Y_all) + 1  # 自動偵測類別數
    Y_all = np.clip(Y_all, 0, num_classes - 1)

    # **使用 GroupShuffleSplit，確保相同 segment 只出現在 train 或 test**
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X_all, Y_all))  # 這樣 `test` 會包含所有類別


    x_train, y_train = X_all[train_idx], Y_all[train_idx]
    x_test, y_test = X_all[test_idx], Y_all[test_idx]

    return x_train, y_train, x_test, y_test, num_classes
def get_data_for_classification(df, sequence_length=32, test_size=0.2):
    """
    讀取 `seatempclass.csv` 並建立符合分類問題的時間窗口數據。
    """
    # 確保 `class` 欄位存在
    if "class" not in df.columns:
        raise ValueError("seatempclass.csv 必須包含 `class` 欄位")

    # 過濾 NaN 資料
    df = df.dropna()

    # 選擇輸入特徵
    feature_columns = [col for col in df.columns if col != "class"]

    # 確保 `df` 至少有 `sequence_length` + 1 筆資料
    if len(df) <= sequence_length:
        raise ValueError(f"資料數量不足，最少需要 {sequence_length + 1} 筆資料")

    # 嘗試用 `GroupShuffleSplit`，但如果 `df` 只有 1 行，則改用 `train_test_split`

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)

    # 產生時間窗口數據
    x_train, y_train = create_time_windows_classification(df_train, feature_columns, sequence_length)
    x_test, y_test = create_time_windows_classification(df_test, feature_columns, sequence_length)

    num_classes = df["class"].nunique()  # 獲取類別數

    return x_train, y_train, x_test, y_test, num_classes

def create_time_windows(df, feature_columns, sequence_length, target_column="SeaTemp_mean"):
    """
    建立時間序列的滑動視窗 (time windows)，輸出 y 為未來 7 天的 sea_temp。

    :param df: DataFrame, 包含所有數據
    :param feature_columns: 需要作為 X 特徵的欄位
    :param sequence_length: 時間序列長度 (32)
    :param target_column: 預測目標欄位，預設為 'sea_temp'
    :return: X (輸入特徵，包含過去的 sea_temp), Y (對應的未來 7 天 sea_temp)
    """
    X, Y = [], []

    # 確保 `X` 包含 `feature_columns` + `sea_temp`
    all_features = feature_columns + [target_column]

    # 依 segment 分組，確保 time window 不跨 segment
    for segment_id, segment_data in df.groupby("segment"):
        segment_data = segment_data.reset_index(drop=True)  # 重設 index 確保連續
        
        for k in range(len(segment_data) - sequence_length - 10 + 1):  # 確保 X 和 Y 長度足夠
            X_seq = segment_data.loc[k:k+sequence_length-1, all_features].values  # **X 包含 sea_temp**
            Y_seq = segment_data.loc[k+sequence_length:k+sequence_length+9, target_column].values  # **Y 只包含未來 7 天 sea_temp**
            
            if X_seq.shape[0] == sequence_length and Y_seq.shape[0] == 10:  # 確保長度符合要求
                X.append(X_seq)
                Y.append(Y_seq)

    return np.array(X), np.array(Y)

def get_data_fixed_gss(df, sequence_length=32, alpha=0.1, test_size=0.2):
    """
    讀取 dailysemtemp.csv，並建立符合公式的時間視窗數據。
    使用 GroupShuffleSplit() 來確保 segment 為單位的訓練/測試集分割。
    """
    # 確保按照時間排序
    df = df.sort_values(by=["segment", "segment_day"]).reset_index(drop=True)

    # 選擇輸入特徵（排除 `sea_temp`, `segment`, `segment_day`）
    feature_columns = [col for col in df.columns if col not in ["SeaTemp_mean", "segment", "segment_day","class"]]
    # feature_columns = [col for col in df.columns if col not in ["segment", "segment_day"]]
    # 確保數據沒有 NaN 值
    df = df.dropna()

    # 指數平滑
    
    print(feature_columns)
    # 使用 GroupShuffleSplit 以 `segment` 來區分訓練與測試集
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["segment"]))
    df = condition_scaler(df, feature_columns)
    df = exponential_smoothing(df, feature_columns, alpha=alpha)
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    # 確保測試集長度足夠
    if len(df_test) < sequence_length + 10:
        sequence_length = max(1, len(df_test) - 10)  # 確保 sequence_length 至少為 1

    # 標準化
    

    # 產生符合 time window 公式的 X, Y
    x_train, y_train = create_time_windows(df_train, feature_columns, sequence_length)
    x_test, y_test = create_time_windows(df_test, feature_columns, sequence_length)

    return x_train, y_train, x_test, y_test

# # 讀取 CSV 並測試
# file_path = "dailysemtemp.csv"
# df = pd.read_csv(file_path)

# # 呼叫函數並獲取處理後的資料
# x_train, y_train, x_test, y_test = get_data_fixed_gss(df)

# # 顯示資料維度
# print("x_train shape:", x_train.shape)  # (樣本數, sequence_length, 特徵數)
# print("y_train shape:", y_train.shape)  # (樣本數, 7) -> 7 天的 sea_temp
# print("x_test shape:", x_test.shape)    # (樣本數, sequence_length, 特徵數)
# print("y_test shape:", y_test.shape)    # (樣本數, 7)
