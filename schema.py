import os
import pandas as pd
import numpy as np
from scipy import stats

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
    # 假設前三欄是 x, y, z 加速度
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    z = df.iloc[:, 2]
    # 計算合力向量
    mag = np.sqrt(x**2 + y**2 + z**2)
    # 將新特徵插入到第四欄 (索引 3)
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
# 1. 特徵與視窗化核心 (保持不變)
# ==========================================================================
def add_magnitude_feature(df):
    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    mag = np.sqrt(x**2 + y**2 + z**2)
    df.insert(3, 'magnitude', mag)
    return df

def create_sliding_windows_with_indices(df, feature_indices, label_index, window_size=128, overlap=64):
    X, y = [], []
    data_values = df.values
    for i in range(0, len(data_values) - window_size, overlap):
        window_features = data_values[i : i + window_size, feature_indices]
        window_labels = data_values[i : i + window_size, label_index]
        mode_result = stats.mode(window_labels, keepdims=True)
        majority_label = int(mode_result.mode[0])
        if majority_label != 0:
            X.append(window_features)
            y.append(majority_label)
    return np.array(X), np.array(y)

# ==========================================================================
# 2. 單一受試者處理管線 (封裝你原本迴圈內的邏輯)
# ==========================================================================
def load_and_preprocess_subject(subject_id, folder_path='data_raw'):
    filename = f"mHealth_subject{subject_id}.log"
    file_path = os.path.join(folder_path, filename)
    
    if not os.path.exists(file_path):
        return None, None

    # 讀取與過濾 Label 0 (對應你原本的 code)
    df = pd.read_csv(file_path, sep='\t', header=None)
    df_filtered = df[df.iloc[:, -1] != 0]

    # 長度檢查
    if len(df_filtered) < 128:
        return None, None

    # 1. 產生包含 magnitude 的資料表 
    df_with_feat = add_magnitude_feature(df_filtered.copy())
    
    # 2. 剪裁出 5 個欄位 (x, y, z, mag, label) 並重新排序 
    # 注意：iloc 之後，label 的相對索引會變成 4
    data_to_process = df_with_feat.iloc[:, [0, 1, 2, 3, -1]]
    
    # 3. 視窗化處理：feature_indices [0,1,2,3], label_index 4 
    X_sub, y_sub = create_sliding_windows_with_indices(
        data_to_process, 
        feature_indices=[0, 1, 2, 3], 
        label_index=4
    )
    return X_sub, y_sub

# ==========================================================================
# 3. 最終整合入口 (完全取代你原本的 glob 迴圈)
# ==========================================================================
def get_final_training_data(folder_path='data_raw'):
    """
    執行 1-10 號受試者的遍歷、處理與最終 np.concatenate 
    回傳與你原本 code 產出完全一致的 X_final, y_final
    """
    all_X = []
    all_y = []
    
    for sid in range(1, 11):
        X_sub, y_sub = load_and_preprocess_subject(sid, folder_path)
        if X_sub is not None and len(X_sub) > 0:
            all_X.append(X_sub)
            all_y.append(y_sub)
            print(f"✅ 受試者 {sid} 處理完成，視窗數量: {len(X_sub)}")
    
    # 執行最終合併 (對應你原本 code 的末端) 
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    
    return X_final, y_final