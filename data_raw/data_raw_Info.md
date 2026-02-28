# mHealth Dataset 資料集說明

## 一、資料集概述
mHealth Dataset 為一個用於人體活動辨識（Human Activity Recognition, HAR）的穿戴式感測器資料集，蒐集 10 位志願者在日常環境下執行多種動作時的身體運動與生理訊號。資料由 University of Granada (UGR) 蒐集，使用 Shimmer2 穿戴式感測器進行量測，取樣頻率為 50 Hz（每秒 50 筆），屬於多變量時間序列資料（Multivariate Time-Series Data）。量測過程未限制受試者動作方式，以模擬真實生活情境，因此具有良好的實務應用價值。

## 二、感測器配置
感測器配戴於三個身體部位，以量測不同肢段的運動行為：
- 胸口（Chest）：量測身體核心運動與姿態變化，並記錄 ECG。
- 右手腕（Right Lower Arm）：量測上肢動作。
- 左腳踝（Left Ankle）：量測步態與下肢運動。

透過多部位同步量測，可完整捕捉人體的整體動態、四肢協調與動作強度差異。

## 三、量測訊號種類
每個感測器皆包含以下三種運動訊號：
- Acceleration（加速度）
- Gyroscope（角速度）
- Magnetometer（磁場方向）

胸口感測器額外提供：
- ECG（雙導程心電圖，單位 mV），主要供未來生理研究使用，通常不納入活動辨識模型。

## 四、活動類別（Activity Labels）
資料集共包含 12 種日常活動：

| Label | 活動內容 | 執行方式 |
|------|-----------|-----------|
| 1 | Standing still | 站立 1 分鐘 |
| 2 | Sitting and relaxing | 坐姿休息 1 分鐘 |
| 3 | Lying down | 平躺 1 分鐘 |
| 4 | Walking | 走路 1 分鐘 |
| 5 | Climbing stairs | 上樓梯 1 分鐘 |
| 6 | Waist bends forward | 彎腰 20 次 |
| 7 | Frontal elevation of arms | 手臂前舉 20 次 |
| 8 | Knees bending (crouching) | 蹲下 20 次 |
| 9 | Cycling | 騎腳踏車 1 分鐘 |
| 10 | Jogging | 慢跑 1 分鐘 |
| 11 | Running | 跑步 1 分鐘 |
| 12 | Jump front & back | 前後跳 20 次 |

Label 0 代表無動作或動作轉換期間（Null Class）。

## 五、資料檔案結構
每位受試者資料儲存為一個獨立檔案：
```
mHealth_subject<SUBJECT_ID>.log
```
每一列代表一個時間點的量測值；由於取樣頻率為 50 Hz，因此每秒包含 50 筆資料。

## 六、欄位定義（24 欄）
資料檔案的每一列包含 24 個欄位，依序如下：

### 胸口感測器（Chest Sensor）
| Column | 說明 | 單位 |
|--------|------|------|
| 1 | Acceleration X | m/s² |
| 2 | Acceleration Y | m/s² |
| 3 | Acceleration Z | m/s² |
| 4 | ECG Lead 1 | mV |
| 5 | ECG Lead 2 | mV |

### 左腳踝感測器（Left Ankle Sensor）
| Column | 說明 | 單位 |
|--------|------|------|
| 6 | Acceleration X | m/s² |
| 7 | Acceleration Y | m/s² |
| 8 | Acceleration Z | m/s² |
| 9 | Gyroscope X | deg/s |
| 10 | Gyroscope Y | deg/s |
| 11 | Gyroscope Z | deg/s |
| 12 | Magnetometer X | local |
| 13 | Magnetometer Y | local |
| 14 | Magnetometer Z | local |

### 右手腕感測器（Right Lower Arm Sensor）
| Column | 說明 | 單位 |
|--------|------|------|
| 15 | Acceleration X | m/s² |
| 16 | Acceleration Y | m/s² |
| 17 | Acceleration Z | m/s² |
| 18 | Gyroscope X | deg/s |
| 19 | Gyroscope Y | deg/s |
| 20 | Gyroscope Z | deg/s |
| 21 | Magnetometer X | local |
| 22 | Magnetometer Y | local |
| 23 | Magnetometer Z | local |

### 活動標籤
| Column | 說明 |
|--------|------|
| 24 | Activity Label（0 為 Null Class） |

## 七、資料單位
- Acceleration：m/s²  
- Gyroscope：deg/s  
- Magnetometer：當地磁場強度（相對值）  
- ECG：mV  

## 八、資料特性
本資料集具有以下特點：
- 多感測器融合（Sensor Fusion）
- 高時間解析度（50Hz）
- 同時涵蓋靜態、動態與功能性動作
- 真實生活場景量測，具良好泛化能力
- 適用於時間序列分析、機器學習與深度學習之人體活動辨識研究

## 九、建議應用方向
- 人體活動辨識（HAR）
- 穿戴式健康監測系統
- 運動行為分類
- 復健與長照分析
- 多感測器資料融合建模
- 時間序列機器學習研究

## 十、引用方式（Citation）
若使用本資料集，請引用以下文獻：

Banos, O., Garcia, R., Holgado-Terriza, J.A., Damas, M.,  
Pomares, H., Rojas, I., Saez, A., Villalonga, C.  
*mHealthDroid: a novel framework for agile development of mobile health applications.*  
Proceedings of the 6th International Work-conference on Ambient Assisted Living and Active Ageing (IWAAL 2014),  
Belfast, United Kingdom, December 2–5, 2014.
