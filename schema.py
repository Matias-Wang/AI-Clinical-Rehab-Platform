import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler


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

# Stage 8：即時串流防護參數
# 注意：進入 RealTimeStreamProcessor 的資料已經過 load_and_preprocess_subject()
# 標準化（Acc/Mag 為受試者級 Z-score、Grav 已 ÷10.0、FFT 已 log1p/5.0），
# 因此此處的邊界檢查採「標準化後尺度」，不可沿用 50 m/s² 這類原始物理門檻。
STREAM_GUARD = {
"N_FEATURES": 8,          # 標準 8D 特徵軸數量，維度不符即視為髒數據
"SANITY_ABS_LIMIT": 50.0,  # Z-score 達 50 個標準差在生理上不可能，判定為硬體雜訊
"DISCONNECT_DIRTY_FRAMES": 25  # 連續 25 幀髒數據（0.5 秒 @50Hz）判定為感測器斷訊
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
def load_mhealth_subject(subject_id, folder_path='data'):
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
def load_and_preprocess_subject(subject_id, folder_path='data'):
    """診斷專用：回傳 (X, y)，並包含物理濾鏡清洗邏輯。
    修改紀錄：
        - 2026/03/07：
            1. 新增物理邊界清洗邏輯，剔除 Sitting (Label 2) 狀態下超過物理門檻的雜訊點。
            2. 修改 load_and_preprocess_subject 函式，整合重力特徵提取]。
        - 2026/03/08：
            1. 新增 Chest Acc X, Y, Z 計算重力分量。
            2. 將重力分量合併入矩陣：[X, Y, Z, Mag, Grav_X, Grav_Y, Grav_Z, Label]。
            3. 將標籤 (y) 放在最後一欄以符合 create_sliding_windows_with_indices 的邏輯。
        - 2026/04/03：
            1. 整合 FFT 頻譜特徵，解決 S2 靜態/動態混淆。
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
    gravity_values = gravity_values / 10.0

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

    # --- 2026/04/03 新增：FFT 特徵廣播 ---
    # 注意：FFT 能量必須用「原始尺度」的 Magnitude 計算（與訓練時一致），
    # 因此受試者級 Z-score 必須排在 FFT 廣播之後，不可提前。
    X_with_fft = []
    for i in range(len(X_sub)):
        window = X_sub[i]
        # 計算該視窗的頻譜能量
        fft_energy = extract_window_fft_energy(window)
        # 建立一個全等的 (128, 1) 矩陣填充該能量值
        fft_feature = np.full((window.shape[0], 1), fft_energy)
        # 合併後變為 8 個特徵
        new_window = np.hstack([window, fft_feature])
        X_with_fft.append(new_window)

    X_with_fft = np.array(X_with_fft)

    # --- 受試者級 Z-score：僅針對 Acc_X, Acc_Y, Acc_Z, Magnitude（前 4 欄）---
    # 用該受試者全部視窗攤平後的 Acc/Mag 值 fit，消除個體強度差異；
    # Gravity（4-6 欄）與 FFT_Energy（第 7 欄）維持原本縮放，不參與此標準化。
    n_windows, n_steps, _ = X_with_fft.shape
    acc_mag_part = X_with_fft[:, :, :4].reshape(-1, 4)
    scaler = StandardScaler()
    scaled_acc_mag = scaler.fit_transform(acc_mag_part).reshape(n_windows, n_steps, 4)
    X_with_fft = np.concatenate([scaled_acc_mag, X_with_fft[:, :, 4:]], axis=2)

    return X_with_fft, y_sub

# ==========================================================================
# 多位受試者處理管線
# ==========================================================================
def get_all_subjects_for_analysis(folder_path='data'):
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
def get_final_training_data(folder_path='data'):
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

# ==========================================================================
# 新增頻譜能量計算函式
# ==========================================================================
def extract_window_fft_energy(window_data):
    """
    計算視窗內加速度量值 (Magnitude) 的頻譜能量
    目的：區分 Sitting (低能量) 與 Waist Bends (高能量)
    """
    # 針對 Magnitude 欄位 (索引 3) 進行 FFT，須為原始尺度（Z-score 之前）
    sig = window_data[:, 3]
    # 執行實數 FFT
    fft_vals = rfft(sig)
    # 計算能量 (去除 DC 分量以集中觀察動作頻率)
    energy = np.sum(np.abs(fft_vals[1:])**2) / len(sig)
    # 縮放：能量值跨度極大，log1p 壓縮後除以 5 對齊到 0~2 區間（SPEC.md §2.5）
    return np.log1p(energy) / 5.0
    
# ==========================================================================
# Stage 6：DTW 距離計算（取代歐幾里德距離）
# ==========================================================================
def dtw_distance(seq_a, seq_b, radius=16):
    """
    計算兩個一維時序訊號的 DTW 距離（Sakoe-Chiba band 限制版）。
    Local cost 採平方差，最終距離開根號，使其與歐幾里德距離同尺度；
    對角線路徑恆為合法解，故 dtw_distance 恆 <= 對應的歐幾里德距離。
    """
    n, m = len(seq_a), len(seq_b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        for j in range(j_start, j_end + 1):
            cost = (seq_a[i - 1] - seq_b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return np.sqrt(D[n, m])


# ==========================================================================
# 新增頻譜能量計算函式
# ==========================================================================
def align_coordinates(X_batch):
    """
    將受試者的感測器座標系對齊至全局重力參考系 (Z-axis)
    X_batch shape: (N_windows, 128, 8)
    """
    # 1. 提取重力分量 (4, 5, 6 欄)
    gravity_data = X_batch[:, :, 4:6+1] # 取得 Grav_X, Grav_Y, Grav_Z
    
    # 2. 計算平均重力方向 (代表受試者的物理基準位姿)
    avg_gravity = np.mean(gravity_data, axis=(0, 1))
    norm = np.linalg.norm(avg_gravity)
    if norm == 0: 
        return X_batch # 避免除以零
    v1 = avg_gravity / norm  # 受試者原始重力向量
    
    # 3. 定義目標向量 (理想的垂直向下，假設為 [0, 0, 1])
    v2 = np.array([0, 0, 1])
    
    # 4. 計算旋轉矩陣 (Rodrigues' Rotation Matrix)
    # 透過外積求旋轉軸，內積求角度
    cross_v = np.cross(v1, v2)
    dot_v = np.dot(v1, v2)
    s = np.linalg.norm(cross_v)
    
    if s < 1e-6: # 若已經對齊，直接回傳
        return X_batch
    
    # 偏對稱矩陣 (Skew-symmetric matrix)
    vx = np.array(
        [[0, -cross_v[2], cross_v[1]],
        [cross_v[2], 0, -cross_v[0]],
        [-cross_v[1], cross_v[0], 0]]
    )
    
    # 旋轉矩陣公式: R = I + vx + vx^2 * ((1 - c) / s^2)
    I = np.eye(3)
    R = I + vx + np.matmul(vx, vx) * ((1 - dot_v) / (s**2))
    
    # 5. 應用旋轉矩陣到 Acc (0,1,2) 與 Grav (4,5,6)
    X_aligned = X_batch.copy()
    
    # 針對每個時間步執行矩陣乘法
    # 為了效能，我們將 (N*128, 3) 進行一次性旋轉
    N, T, F = X_batch.shape
    
    # 校準加速度
    acc_flat = X_aligned[:, :, 0:3].reshape(-1, 3)
    X_aligned[:, :, 0:3] = np.dot(acc_flat, R.T).reshape(N, T, 3)
    
    # 校準重力分量
    grav_flat = X_aligned[:, :, 4:7].reshape(-1, 3)
    X_aligned[:, :, 4:7] = np.dot(grav_flat, R.T).reshape(N, T, 3)
    
    return X_aligned


# ==========================================================================
# 黃金範本擷取
# ==========================================================================
def extract_golden_template(X, y, target_label=7):
    """從指定受試者的 (X, y) 中擷取第一個符合 target_label 的視窗作為黃金範本。

    Parameters
    ----------
    X : numpy.ndarray
        shape (n_windows, 128, 8)，來自 load_and_preprocess_subject()。
    y : numpy.ndarray
        shape (n_windows,)，對應每個視窗的多數投票標籤。
    target_label : int
        要擷取的目標動作標籤（預設 7：Frontal Elevation of Arms）。

    Returns
    -------
    numpy.ndarray
        shape (128, 8) 的單一視窗，作為黃金範本。
    """
    indices = np.where(y == target_label)[0]
    if len(indices) == 0:
        raise ValueError(f"找不到標籤為 {target_label} 的視窗，無法擷取黃金範本。")
    return X[indices[0]]


# ==========================================================================
# 臨床品質檢驗
# ==========================================================================

import numpy as np
from collections import deque


# ==========================================================================
# Stage 8：臨床串流例外型別
# ==========================================================================
class SensorStreamError(Exception):
    """所有即時串流層例外的基底類別。"""


class DirtyFrameError(SensorStreamError):
    """單一影格未通過完整性檢查（NaN／Inf／維度錯誤／數值超出生理範圍）。"""


class SensorDisconnectedError(SensorStreamError):
    """連續髒影格數超過門檻，判定感測器已斷訊。"""


# --- Week 4: 臨床品質閘門 (Generation 4) ---
class ClinicalQualityGate:
    def __init__(self, golden_template):
        self.GOLDEN_VAR_LIMIT = 0.001595  # S10 標竿
        self.MIN_SAFE_LIMIT = 0.0005     # 邊緣案例 (S7/S9) 及格線
        self.template = golden_template
        
    def get_quality_report(self, X_window):
        """
        針對單一或多個 128 步視窗進行品質診斷
        """
        # 修正維度處理，確保針對視窗內的 Y 軸變異數進行計算
        if X_window.ndim == 3:
            var_y = np.mean(np.var(X_window[:, :, 5], axis=1))
        else:
            var_y = np.var(X_window[:, 5])

        # Stage 8 修正：NaN/Inf 防護（fail-safe）。
        # 原本缺少此檢查時，含 NaN 的視窗會使 var_y 變成 NaN，而
        # `NaN < MIN_SAFE_LIMIT` 恆為 False，導致髒數據被判定為「品質良好」
        # 並直接送入模型推論。臨床上必須採取「寧可不輸出，也不輸出錯誤結果」。
        if not np.isfinite(var_y):
            return False, 0.0, "⚠️ 訊號含無效值 (NaN/Inf)，AI 停止預測。"

        if var_y < self.MIN_SAFE_LIMIT:
            score = (var_y / self.MIN_SAFE_LIMIT) * 60
            return False, score, "⚠️ 動作幅度嚴重不足，AI 停止預測。"
        
        score = min(100, (var_y / self.GOLDEN_VAR_LIMIT) * 100)
        return True, score, "✅ 數據品質良好，正在辨識中。"


# ==========================================================================
# 即時數據處理
# ==========================================================================
class RealTimeStreamProcessor:
    """滑動視窗緩衝區，含 Stage 8 的髒數據攔截與緩衝區復原機制。"""

    def __init__(
        self,
        window_size: int = 128,
        stride: int = 64,
        sanity_abs_limit: float = STREAM_GUARD["SANITY_ABS_LIMIT"],
        disconnect_dirty_frames: int = STREAM_GUARD["DISCONNECT_DIRTY_FRAMES"],
    ) -> None:
        """初始化緩衝區與防護統計量。

        Parameters
        ----------
        window_size : int
            視窗長度（時間步數），預設 128。
        stride : int
            視窗輸出步長，預設 64（50% 重疊）。
        sanity_abs_limit : float
            標準化尺度下的絕對值上限，超過即判定為硬體雜訊。
        disconnect_dirty_frames : int
            連續髒影格達此數量時判定為感測器斷訊。
        """
        self.window_size = window_size
        self.stride = stride
        self.sanity_abs_limit = sanity_abs_limit
        self.disconnect_dirty_frames = disconnect_dirty_frames
        self.buffer = deque(maxlen=window_size)
        self.new_data_counter = 0

        # Stage 8：串流健康度統計（供壓力測試與運維監控使用）
        self.total_frames = 0
        self.dirty_frames = 0
        self.consecutive_dirty = 0
        self.buffer_resets = 0

    def validate_frame(self, sensor_row) -> np.ndarray:
        """檢查單一影格的完整性，通過則回傳標準化後的 float64 陣列。

        Parameters
        ----------
        sensor_row : array_like
            單一時間步的感測器資料，預期長度為 8。

        Returns
        -------
        numpy.ndarray
            shape (8,) 的 float64 影格。

        Raises
        ------
        DirtyFrameError
            影格無法轉為數值、維度不符、含 NaN/Inf，或數值超出生理合理範圍。
        """
        try:
            frame = np.asarray(sensor_row, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise DirtyFrameError(f"影格無法轉換為數值：{e}") from e

        if frame.shape != (STREAM_GUARD["N_FEATURES"],):
            raise DirtyFrameError(
                f"影格維度錯誤：期望 ({STREAM_GUARD['N_FEATURES']},)，實得 {frame.shape}"
            )

        if not np.all(np.isfinite(frame)):
            raise DirtyFrameError("影格含 NaN 或 Inf，判定為感測器故障或封包毀損。")

        if np.any(np.abs(frame) > self.sanity_abs_limit):
            peak = float(np.max(np.abs(frame)))
            raise DirtyFrameError(
                f"影格數值 {peak:.2f} 超出合理範圍 "
                f"(|x| > {self.sanity_abs_limit})，判定為硬體雜訊。"
            )

        return frame

    def reset_buffer(self) -> None:
        """清空緩衝區與步長計數器。

        Stage 8 新增：髒影格一旦進入 deque，後續最多 window_size 步的視窗都會
        被污染。攔截到髒數據時必須丟棄整段緩衝，等待重新累積乾淨的完整視窗，
        避免系統輸出「看起來正常但基於污染資料」的評分。
        """
        self.buffer.clear()
        self.new_data_counter = 0
        self.buffer_resets += 1

    def push_data(self, sensor_row):
        """
        將單一時間步的感測器數據推入緩衝區

        Raises
        ------
        DirtyFrameError
            影格未通過 validate_frame() 檢查，緩衝區已同步清空。
        SensorDisconnectedError
            連續髒影格數達到 disconnect_dirty_frames 門檻。
        """
        self.total_frames += 1

        try:
            frame = self.validate_frame(sensor_row)
        except DirtyFrameError:
            # 髒影格不入列，並清空既有緩衝以免污染後續視窗
            self.dirty_frames += 1
            self.consecutive_dirty += 1
            self.reset_buffer()
            if self.consecutive_dirty >= self.disconnect_dirty_frames:
                raise SensorDisconnectedError(
                    f"連續 {self.consecutive_dirty} 幀無效資料，判定感測器斷訊。"
                ) from None
            raise

        self.consecutive_dirty = 0
        self.buffer.append(frame)
        self.new_data_counter += 1

        # 達到步長且緩衝區已滿，回傳視窗數據進行分析
        if self.new_data_counter >= self.stride and len(self.buffer) == self.window_size:
            self.new_data_counter = 0
            return True, np.array(self.buffer)
        return False, None

    def get_health_stats(self) -> dict:
        """回傳串流健康度統計，供壓力測試與長時間運行監控使用。

        Returns
        -------
        dict
            包含總影格數、髒影格數、髒數據比例、緩衝區重置次數與目前緩衝長度。
        """
        dirty_ratio = self.dirty_frames / self.total_frames if self.total_frames else 0.0
        return {
            "total_frames": self.total_frames,
            "dirty_frames": self.dirty_frames,
            "dirty_ratio": dirty_ratio,
            "buffer_resets": self.buffer_resets,
            "consecutive_dirty": self.consecutive_dirty,
            "buffer_len": len(self.buffer),
        }

class RealTimeBiofeedbackEngine(RealTimeStreamProcessor):
    def __init__(self, quality_gate, golden_template, model, window_size=128, stride=64, dtw_radius=16):
        super().__init__(window_size, stride)
        self.gate = quality_gate
        self.golden_template = golden_template
        self.model = model
        self.dtw_radius = dtw_radius

    def calculate_similarity(self, current_window):
        """
        [物理特徵比對] 依照 SPEC 8.5 節實作相似度映射
        Stage 6：改用 DTW 取代歐幾里德距離，容忍動作節奏（相位）差異。
        """
        current_y = current_window[:, 5] # Grav_Y
        golden_y = self.golden_template[:, 5]

        distance = dtw_distance(current_y, golden_y, radius=self.dtw_radius)

        # 評分公式：$Score_{sim} = \max(0, 100 - Distance \times 15)$
        # (這裡微調係數為 15 以提升實時顯示的寬容度)
        sim_score = max(0, 100 - (distance * 15))
        return sim_score

    def _build_halt_result(self, reason: str, msg: str) -> dict:
        """組出髒數據／斷訊情境下的紅燈回傳結果。

        Parameters
        ----------
        reason : str
            攔截原因代碼（DIRTY_DATA 或 DISCONNECTED）。
        msg : str
            要顯示於前端的說明文字。

        Returns
        -------
        dict
            與 process_live_frame() 一致的結果格式，status 固定為 HALT。
        """
        return {
            "status": "HALT",
            "ui_color": "RED",
            "msg": msg,
            "score": 0.0,
            "similarity": 0,
            "predict_label": None,
            "reason": reason,
        }

    def process_live_frame(self, new_frame):
        # Stage 8：髒數據／斷訊攔截。push_data() 已在拋出例外前清空緩衝區，
        # 此處只需將其轉為前端可理解的紅燈狀態，讓長時間串流不會因單一
        # 壞封包而中斷。為避免雜訊期間洪水式推播，僅在「乾淨→髒」的狀態
        # 轉換點與斷訊升級時輸出。
        try:
            ready, window_data = self.push_data(new_frame)
        except SensorDisconnectedError as e:
            return self._build_halt_result("DISCONNECTED", f"🔌 感測器斷訊：{e}")
        except DirtyFrameError as e:
            if self.consecutive_dirty == 1:
                return self._build_halt_result("DIRTY_DATA", f"⚠️ 訊號異常：{e}")
            return None

        if not ready: return None

        # 1. 執行 Week 4 品質檢查
        is_valid, q_score, q_msg = self.gate.get_quality_report(window_data)
        
        if not is_valid:
            return {
                "status": "HALT", 
                "ui_color": "RED", 
                "msg": q_msg, 
                "score": q_score,
                "similarity": 0,
                "predict_label": None,
                "reason": "LOW_QUALITY",
            }

        # 2. 執行 Week 3 (v3) 模型推論
        input_data = np.expand_dims(window_data, axis=0) # 符合 (1, 128, 8)
        prediction = self.model.predict(input_data, verbose=0)
        predict_label = np.argmax(prediction)
        
        # 3. 相似度與綜合評分 (0.4 品質 + 0.6 姿勢)
        sim_score = self.calculate_similarity(window_data)
        final_score = (q_score * 0.4) + (sim_score * 0.6)
        
        # 4. UI 顏色狀態機
        ui_color = "GREEN" if sim_score > 35 else "YELLOW"
        msg = f"✅ AI 辨識：{ACTIVITY_LABELS.get(predict_label, 'Unknown')}" #
        
        return {
            "status": "PROCEED", 
            "ui_color": ui_color, 
            "msg": msg, 
            "score": final_score,
            "similarity": sim_score,
            "predict_label": predict_label,
            "reason": "OK",
        }