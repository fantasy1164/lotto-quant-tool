# Lotto Quant Tool 🎲📊

以台灣彩券歷史開獎資料為基礎，結合統計特徵、Random Forest（隨機森林）與前端資料視覺化的機率研究工具。

本專案主要用於觀察歷史開獎資料中的號碼出現頻率、遺漏期數、近期冷熱狀態，以及示範如何將歷史資料轉換為機器學習特徵並進行分類模型分析。

> [!WARNING]
> **本專案僅供機率、統計、資料分析、機器學習與程式技術研究／教育用途。**  
> 彩券開獎具有隨機性，歷史資料、冷熱號、遺漏期數、模型機率或任何演算法輸出，**均無法保證或證明可以預測未來開獎結果，也不會提高實際中獎機率**。  
> 本專案提供的任何號碼、排名、機率或分析結果皆**不構成投注、理財或獲利建議**。請勿依賴本工具進行投注決策，並請遵守所在地法律、投注年齡限制及量力而為。

---

## 專案功能

- 自動取得台灣彩券歷史開獎資料
- 使用最近約 100 期資料建立分析樣本
- 將近期開獎狀態轉換為 0 / 1 特徵矩陣
- 使用 `RandomForestClassifier` 建立分類模型
- 計算各號碼的模型分數／機率輸出
- 顯示近期 20 期出現次數（冷／熱狀態）
- 顯示目前遺漏期數
- 顯示 Random Forest Feature Importance
- 計算下一期預定開獎日期
- 透過 GitHub Actions 定期更新 `data.json`
- 使用純 HTML / CSS / JavaScript 呈現 Dashboard

---

## 支援彩券

| 彩券 | 分析方式 | 備註 |
|---|---|---|
| 大樂透 | Random Forest + 歷史資料 | 第一區 1～49，取 6 號 |
| 威力彩 | Random Forest + 歷史資料 | 第一區 1～38，取 6 號；第二區以歷史資料統計顯示 |
| 今彩 539 | Random Forest + 歷史資料 | 1～39，取 5 號 |
| 3星彩 | 模擬資料展示 | **目前不是以真實歷史資料訓練** |
| 4星彩 | 模擬資料展示 | **目前不是以真實歷史資料訓練** |

> [!IMPORTANT]
> 目前 `3星彩` 與 `4星彩` 的結果由程式以隨機方式產生，主要用於 UI 與資料流程展示，不應解讀為模型預測結果。

---

## 分析方法

目前主要模型為：

```text
Random Forest Classifier
n_estimators = 100
random_state = 42
```

每個號碼會根據最近 5 期是否出現建立特徵：

```text
t-1
t-2
t-3
t-4
t-5
最近 5 期出現次數（freq）
```

概念上會形成：

```text
歷史開獎資料
      ↓
號碼 0 / 1 矩陣
      ↓
最近 5 期特徵
      ↓
Random Forest
      ↓
各號碼模型輸出分數
      ↓
排序 + 遺漏期 + 冷熱狀態
      ↓
data.json
      ↓
Web Dashboard
```

### Feature Importance

Dashboard 同時顯示模型對各項特徵的相對重要程度，例如：

- 前 1 期是否出現
- 前 2 期是否出現
- 前 3 期是否出現
- 前 4 期是否出現
- 前 5 期是否出現
- 最近 5 期總出現次數

Feature Importance 只代表模型在目前訓練資料中的特徵使用情況，**不代表該特徵具有因果關係或真正的開獎預測能力**。

---

## 專案結構

```text
lotto-quant-tool/
├── .github/
│   └── workflows/
│       └── daily_update.yml   # GitHub Actions 自動更新
├── data.json                  # 前端顯示用分析結果
├── index.html                 # Dashboard
├── update_data.py             # 資料取得、特徵工程與模型分析
└── README.md
```

---

## 執行環境

建議：

- Python 3.10+
- NumPy
- Requests
- scikit-learn

安裝相依套件：

```bash
pip install requests numpy scikit-learn
```

---

## 本機執行

### 1. Clone repository

```bash
git clone https://github.com/fantasy1164/lotto-quant-tool.git
cd lotto-quant-tool
```

### 2. 更新分析資料

```bash
python update_data.py
```

執行完成後會更新：

```text
data.json
```

### 3. 啟動本機 Web Server

由於前端會透過 `fetch('data.json')` 讀取資料，不建議直接以 `file://` 開啟 `index.html`。

可以使用 Python 內建 HTTP Server：

```bash
python -m http.server 8000
```

瀏覽：

```text
http://localhost:8000
```

---

## 自動更新

專案包含 GitHub Actions：

```text
.github/workflows/daily_update.yml
```

目前排程：

```cron
30 13 * * *
```

也就是每天約：

```text
UTC 13:30
台灣時間 UTC+8 約 21:30
```

Workflow 會依序：

```text
Checkout repository
       ↓
Setup Python 3.10
       ↓
Install dependencies
       ↓
python update_data.py
       ↓
產生 / 更新 data.json
       ↓
Commit & Push
```

也支援從 GitHub Actions 頁面使用 `workflow_dispatch` 手動執行。

---

## 資料來源與 Fallback 行為

程式會嘗試透過台灣彩券公開 API 取得歷史開獎資料：

```text
https://api.taiwanlottery.com.tw/api/v1/lottery
```

若 API 請求失敗或執行環境無法取得資料，程式目前會啟用 fallback 機制，建立模擬歷史資料，讓模型與 Dashboard 仍可正常執行。

因此使用分析結果前，請特別注意：

> **Dashboard 正常顯示，不代表該次資料一定來自真實台灣彩券歷史資料。**

若要進行正式統計研究，建議另外記錄每次更新的資料來源狀態，或修改程式使 API 失敗時直接停止執行，以避免將模擬資料與真實資料混用。

---

## 如何解讀模型輸出

Dashboard 中顯示的「機率」或排序結果，是模型基於目前訓練資料與特徵所產生的分類輸出。

它應該被理解為：

```text
「模型在這組歷史資料與特徵設計下給出的分數」
```

而不是：

```text
「下一期真正開出該號碼的客觀機率」
```

在公平且獨立的彩券機制中，每一期開獎不會因為某號碼過去很久沒出現、近期很熱門，或模型給予較高分數，就因此必然提高下一期被抽中的機率。

---

## 研究限制

目前模型具有幾項明顯限制：

1. 主要僅使用近期開獎與出現頻率相關特徵。
2. 訓練資料量約為近期 100 期，樣本規模有限。
3. 沒有進行嚴格的時間序列 out-of-sample backtest。
4. 沒有證明模型相較於隨機選號具有統計顯著優勢。
5. API 失敗時可能使用模擬 fallback 資料。
6. 3星彩與 4星彩目前為模擬展示資料。
7. 模型輸出的分類機率不等同於真實開獎機率。

若作為機器學習研究專案，後續可加入：

- Walk-forward validation
- Rolling-window backtest
- Random baseline comparison
- Calibration curve
- Brier score
- Log loss
- Monte Carlo simulation
- 不同 lookback window 比較
- 模型穩定性分析
- 真實資料／模擬資料來源標記

---

## 免責聲明

本專案與台灣彩券股份有限公司及其他彩券發行、經銷或主管單位**無任何官方隸屬、合作、背書或推薦關係**。

本專案作者與貢獻者不保證：

- 資料永遠正確、完整或即時
- API 永遠可用
- 模型輸出具有實際預測能力
- 使用本工具可以提高中獎機率
- 使用本工具可以獲得任何財務利益

使用者應自行判斷資料與程式的適用性，並自行承擔使用本專案所產生的任何結果與風險。

**再次提醒：本專案僅供機率研究、統計分析、資料視覺化、機器學習實驗及程式教育用途，請勿將任何模型輸出視為投注建議。**

---

## Responsible Use

如果你選擇參與任何合法彩券或博彩活動，請務必：

- 僅使用可承受損失的娛樂預算
- 不借貸、不追損、不因模型結果增加投注金額
- 不將機器學習輸出誤認為必勝策略
- 遵守所在地相關法律與年齡限制

彩券的本質仍是隨機遊戲；這個 repository 的價值在於 **資料工程、統計思考與機器學習實驗**，而不是提供中獎保證。
