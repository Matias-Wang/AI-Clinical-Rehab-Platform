# 臨床復健動作辨識：技術架構定義 (Schema Definitions)

此文件定義了 MHEALTH 數據集的欄位索引、動作標籤以及臨床物理常數，作為 AI 模型訓練與稽核的基準。

## 1. 數據欄位索引 (MHEALTH_COLUMNS)
定義原始數據集中的關鍵特徵位置：
* **LABEL**: 索引 `23` (動作標籤列)
* **LEFT_ANKLE_ACC_X**: 索引 `5` (左腳踝加速度 X 軸)
* **LEFT_ANKLE_ACC_Y**: 索引 `6` (左腳踝加速度 Y 軸)
* **LEFT_ANKLE_ACC_Z**: 索引 `7` (左腳踝加速度 Z 軸)

## 2. 動作標籤定義 (ACTIVITY_LABELS)
定義數值標籤與臨床動作的對應關係：
* **1**: Standing Still (靜止站立)
* **2**: Sitting and relaxing (坐姿休息)
* **3**: Lying down (平躺)
* **4**: Walking (行走)
* **5**: Climbing stairs (爬樓梯)
* **6**: Waist bends forward (腰部前彎)
* **7**: Frontal arms elevation (手臂前舉)
* **8**: Knees bending (crouching) (深蹲/彎膝)
* **9**: Cycling (騎腳踏車)
* **10**: Jogging (慢跑)
* **11**: Running (跑步)
* **12**: Jump front & back (前後跳躍)

## 3. 物理與安全常數 (PHYSICAL_CONSTANTS)
定義數據稽核與臨床安全護欄的門檻值：
* **GRAVITY**: `9.80665` (地球重力常數，用於靜止校準基準)
* **ACC_ERROR_THRESHOLD**: `50.0` (物理上限攔截點，防止讀取到磁力計或感測器碰撞)
* **CALIBRATION_RANGE**: `9.0` ~ `11.0` (標籤 1 容許的重力誤差範圍)
* **KINETIC_GATE_WALKING**: `> 11.0` (標籤 4 行走狀態應具備的最低動能)

## 4. AI 模型參數 (MODEL_HYPERPARAMETERS)
* **WINDOW_SIZE**: `128` (視窗長度：2.56 秒)
* **OVERLAP**: `64` (重疊長度：50%)
* **SAMPLING_RATE**: `50Hz` (每秒採樣 50 次)