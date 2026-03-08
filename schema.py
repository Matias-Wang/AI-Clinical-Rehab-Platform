import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt


# 欄位索引定義 (0-based index)
MHEALTH_COLUMNS = {
"CHEST_ACC_X": 0,
"CHEST_ACC_Y": 1,
"CHEST_ACC_Z": 2,
"ECG_LEAD_1": 3,
"ECG_LEAD_2": 4,
"LEFT_ANKLE_ACC_X": 5,
"LEFT_ANKLE_ACC_Y": 6,
"LEFT_ANKLE_ACC_Z": 7,
"LEFT_ANKLE_GYRO_X": 8,
"LEFT_ANKLE_GYRO_Y": 9,
"LEFT_ANKLE_GYRO_Z": 10,
"LEFT_ANKLE_MAG_X": 11,
"LEFT_ANKLE_MAG_Y": 12,
"LEFT_ANKLE_MAG_Z": 13,
"RIGHT_ARM_ACC_X": 14,
"RIGHT_ARM_ACC_Y": 15,
"RIGHT_ARM_ACC_Z": 16,
"LABEL": 23
}

# 物理臨床變數
PHYSICAL_CONSTANTS = {
"GRAVITY": 9.80665,
"ACC_ERROR_THRESHOLD": 50.0,
"SAMPLING_RATE_HZ": 50
}

"""
數據稽核清單：
靜止重力門檻 (9.0 到 11.0)：正常靜止，可以用來校準感測器。
原因：確認單位正確。地球重力是 9.8。如果數據在這個區間，代表你的單位是 m/s2。如果只有 1 左右，代表單位是 g。如果超過 100，代表你讀到磁力計了。

人體物理極限 (最高 50.0)：正常運動，這是 AI 辨識的主力數據。
原因：過濾雜訊。一般復健動作很少超過 5g (約 50 m/s2)。如果數據突然噴到 100，那通常是感測器撞到東西或是電子雜訊，這種「髒數據」絕對不能讓 AI 學習。

運動判定門檻 (大於 11.0)：
原因：區分靜止與動作。走路時會有衝擊力，合成加速度一定會高於 9.8。如果數值一直維持在 9.8 附近，代表患者根本沒在動，只是坐著搖晃。

數值 超過 50：異常數據，系統應自動剔除。
數值 低於 9：單位錯誤或感測器失效。
"""


# 動作標籤映射
ACTIVITY_LABELS = {
1: "Standing Still",
2: "Sitting and Relaxing",
3: "Lying Down",
4: "Walking",
5: "Climbing Stairs",
6: "Waist Bends Forward",
7: "Frontal Elevation of Arms",
8: "Knees Bending (Crouching)",
9: "Cycling",
10: "Jogging",
11: "Running",
12: "Jump Front & Back"
}

# ==========================================================================
# 計算合力向量 (Magnitude) 的邏輯
# ==========================================================================
def add_magnitude_feature(df):
    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    mag = np.sqrt(x**2 + y**2 + z**2)
    df.insert(3, 'magnitude', mag)
    return df

# ==========================================================================
# 滑動視窗實作函數
# ==========================================================================
def create_sliding_windows_with_indices(
        df, 
        feature_indices, 
        label_index, 
        window_size = 128, 
        overlap = 64):
    X = []
    y = []
    data_values = df.values

    for i in range(0, len(data_values) - window_size, overlap):
        # 根據你傳入的索引抓取特徵
        window_features = data_values[i : i + window_size, feature_indices]
        # 根據你傳入的索引抓取標籤
        window_labels = data_values[i : i + window_size, label_index]
        
        mode_result = stats.mode(window_labels, keepdims=True)
        majority_label = int(mode_result.mode[0])
        
        if majority_label != 0:
            X.append(window_features)
            y.append(majority_label)
            
    return np.array(X), np.array(y)

# ==========================================================================
# 基本單一受試者資料讀取函數
# ==========================================================================
def load_mhealth_subject(subject_id, folder_path='data_raw'):
    """原子化讀取：負責最基礎的檔案讀取與錯誤檢查"""
    filename = f"mHealth_subject{subject_id}.log"
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        print(f"❌ 錯誤：找不到檔案 {file_path}")
        return None
    return pd.read_csv(file_path, sep='\t', header=None)





# ==========================================================================
# 2. 單一受試者處理管線
# ==========================================================================
def load_and_preprocess_subject(subject_id, folder_path='data_raw'):
    """診斷專用：回傳 (X, y)，並包含物理濾鏡清洗邏輯。
    修改紀錄：
        - 2026/03/07：
            1. 新增物理邊界清洗邏輯，剔除 Sitting (Label 2) 狀態下超過物理門檻的雜訊點。
            2. 修改 load_and_preprocess_subject 函式，整合重力特徵提取]。
    """
    df = load_mhealth_subject(subject_id, folder_path) 
    if df is None: 
        return None, None
    
    df_filtered = df[df.iloc[:, -1] != 0]
    # 長度檢查
    if len(df_filtered) < 128:
        return None, None

    # 1. 產生包含 magnitude 的資料表 
    df_with_feat = add_magnitude_feature(df_filtered.copy())

    # 2. 物理邊界清洗邏輯 (2026/03/07)
    upper_bound = PHYSICAL_CONSTANTS["GRAVITY"] + 1.0
    noise_mask = (df_with_feat.iloc[:, -1] == 2) & (df_with_feat['magnitude'] > upper_bound)
    df_with_feat = df_with_feat[~noise_mask]

    # 3. --- 2026/03/08 新增：重力特徵分離 ---
    # 提取 Chest Acc X, Y, Z 計算重力分量
    acc_raw = df_with_feat.iloc[:, :3].values
    gravity_values = apply_low_pass_filter(acc_raw) 

    # 將重力分量合併入矩陣：[X, Y, Z, Mag, Grav_X, Grav_Y, Grav_Z, Label]
    # 將標籤 (y) 放在最後一欄以符合 create_sliding_windows_with_indices 的邏輯
    combined_matrix = np.hstack([
        df_with_feat.iloc[:, [0, 1, 2, 3]].values, # X, Y, Z, Magnitude
        gravity_values,                            # Grav_X, Grav_Y, Grav_Z
        df_with_feat.iloc[:, -1].values.reshape(-1, 1) # Label
    ])

     # ---------------------------------------
    
    # 2. 剪裁出 5 個欄位 (x, y, z, mag, label) 並重新排序 
    # 注意：iloc 之後，label 的相對索引會變成 4
    # data_to_process = df_with_feat.iloc[:, [0, 1, 2, 3, -1]]

    # 轉回 DataFrame 以供後續視窗化函數讀取 .values
    data_to_process = pd.DataFrame(combined_matrix)
    
    # 4. 視窗化處理：feature_indices [0,1,2,3], label_index 4 
    X_sub, y_sub = create_sliding_windows_with_indices(
        data_to_process, 
        # feature_indices=[0, 1, 2, 3], 
        # label_index = 4
        feature_indices=[0, 1, 2, 3, 4, 5, 6],     # X, Y, Z, Mag, Grav_X, Grav_Y, Grav_Z
        label_index = 7
    )
    return X_sub, y_sub

# ==========================================================================
# 多位受試者處理管線
# ==========================================================================
def get_all_subjects_for_analysis(folder_path='data_raw'):
    """
    診斷專用：回傳字典 {sid: (X, y)}。
    保留受試者獨立性，讓你能量化 Subject 1 vs Subject 4 的差異。
    """
    all_data = {}
    for sid in range(1, 11):
        X_s, y_s = load_and_preprocess_subject(sid, folder_path)
        if X_s is not None:
            all_data[sid] = (X_s, y_s)
    return all_data


# ==========================================================================
# 最終整合入口
# ==========================================================================
def get_final_training_data(folder_path='data_raw'):
    """
    任務 B：模型訓練專用。
    回傳合併後的 X_final, y_final (與你原本的 code 產出一致)。
    """
    # 先取得個別受試者的字典
    data_dict = get_all_subjects_for_analysis(folder_path)
    # 提取所有 X 和 y 並合併
    all_X = [v[0] for v in data_dict.values()]
    all_y = [v[1] for v in data_dict.values()]
    
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    return X_final, y_final

# ==========================================================================
# 物理濾波器實作：Butterworth 低通濾波器
# ==========================================================================
def apply_low_pass_filter(data, cutoff=0.3, fs=50, order=2):
    """
    使用 Butterworth 低通濾波器分離重力分量
    - cutoff: 截止頻率 (Hz)，人體靜止重力特徵通常低於 0.3Hz
    - fs: mHealth 資料集的取樣頻率 (50Hz)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # 沿著時間軸 (axis=0) 進行濾波
    # gravity = lfilter(b, a, data, axis=0)
    # 使用 filtfilt 替代 lfilter，消除時間延遲
    gravity = filtfilt(b, a, data, axis=0)
    return gravity