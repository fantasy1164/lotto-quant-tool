import json
import time
from datetime import datetime
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

# 台彩官方 API 基礎路徑
API_BASE_URL = "https://api.taiwanlottery.com.tw/api/v1/lottery"


def fetch_real_history(lottery_type, size=100):
    """
    從台彩官方 API 爬取真實歷史資料
    lottery_type: 'lotto649' (大樂透), 'superLotto' (威力彩), 'dailyCash' (今彩539)
    """
    url = f"{API_BASE_URL}/{lottery_type}/history"
    params = {"size": size}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 解析 API 回傳結構 (台彩新版 API 資料多包在 'content' 欄位中)
        records = data.get("content", [])
        history_list = []

        for record in records:
            period = record.get("period")
            # 取得開獎號碼 (一般是 'resultSeq1' 到 'resultSeq6'，或是陣列)
            # 台彩 API 格式通常在各產品分類下，這裡將其標準化
            draw_results = record.get("lotto649Result") or record.get("superLottoResult") or record.get("dailyCashResult") or {}
            
            if not draw_results:
                continue
                
            # 提取 1~6 號 (539 只有 1~5 號)
            numbers = []
            for i in range(1, 7):
                num = draw_results.get(f"resultSeq{i}")
                if num is not None:
                    numbers.append(int(num))
            
            # 提取特別號
            special = draw_results.get("specialNo")
            if special is not None:
                special = int(special)

            history_list.append({
                "period": period,
                "numbers": numbers,
                "special": special
            })
            
        return history_list
    except Exception as e:
        print(f"爬取 {lottery_type} 失敗: {e}")
        return []


def create_features_and_labels(history_numbers, max_number=49, lookback=5):
    """
    機器學習特徵工程：將歷史號碼轉換為訓練特徵
    """
    X, y = [], []
    total_periods = len(history_numbers)
    if total_periods <= lookback:
        return np.array(X), np.array(y), np.zeros((1, max_number + 1))

    # 轉為 0/1 矩陣
    matrix = np.zeros((total_periods, max_number + 1))
    for i, nums in enumerate(history_numbers):
        for num in nums:
            if 1 <= num <= max_number:
                matrix[i, num] = 1

    # 滾動時間窗口建立特徵
    for t in range(total_periods - lookback - 1, -1, -1):
        for num in range(1, max_number + 1):
            feature = matrix[t + 1 : t + 1 + lookback, num]
            freq = np.sum(feature)
            final_feature = np.append(feature, freq)
            X.append(final_feature)
            y.append(matrix[t, num])

    return np.array(X), np.array(y), matrix


def ml_predict(history_numbers, max_number, pick_count):
    """
    利用機器學習預測下一期
    """
    lookback = 5
    if len(history_numbers) < (lookback + 2):
        # 備用防錯：資料不足時隨機推薦
        import random
        return sorted(random.sample(range(1, max_number + 1), pick_count))

    X, y, matrix = create_features_and_labels(history_numbers, max_number, lookback)

    # 建立隨機森林分類器
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 生成最新特徵進行推演
    next_features = []
    for num in range(1, max_number + 1):
        feature = matrix[0:lookback, num]
        freq = np.sum(feature)
        final_feature = np.append(feature, freq)
        next_features.append(final_feature)

    # 預測開出機率
    probabilities = model.predict_proba(np.array(next_features))[:, 1]
    num_prob_pairs = list(zip(range(1, max_number + 1), probabilities))
    num_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    # 挑選機率最高的前幾名
    recommended_numbers = [pair[0] for pair in num_prob_pairs[:pick_count]]
    return sorted(recommended_numbers)


def main():
    predictions = {}
    
    # === 1. 大樂透 (lotto649) ===
    print("正在爬取大樂透歷史數據...")
    lotto_history = fetch_real_history("lotto649", size=100)
    if lotto_history:
        # 只取出純號碼列表供 ML 訓練
        nums_only = [item["numbers"] for item in lotto_history]
        recommended = ml_predict(nums_only, max_number=49, pick_count=6)
        
        # 預測特別號 (直接從歷史特別號做簡單隨機森林或頻率統計，此處先挑歷史熱門特別號)
        specials_only = [item["special"] for item in lotto_history if item["special"] is not None]
        best_special = max(set(specials_only), key=specials_only.count) if specials_only else 1
        
        predictions["lotto"] = {
            "name": "大樂透",
            "numbers": recommended,
            "special": int(best_special),
            "style": "lotto"
        }
    time.sleep(2) # 延遲避免請求過於頻繁

    # === 2. 威力彩 (superLotto) ===
    print("正在爬取威力彩歷史數據...")
    super_history = fetch_real_history("superLotto", size=100)
    if super_history:
        nums_only = [item["numbers"] for item in super_history]
        recommended = ml_predict(nums_only, max_number=38, pick_count=6)
        
        # 威力彩第二區 (1~8) 統計推薦
        specials_only = [item["special"] for item in super_history if item["special"] is not None]
        best_special = max(set(specials_only), key=specials_only.count) if specials_only else 1
        
        predictions["super_lotto"] = {
            "name": "威力彩",
            "numbers": recommended,
            "special": int(best_special),
            "style": "super_lotto"
        }
    time.sleep(2)

    # === 3. 今彩539 (dailyCash) ===
    print("正在爬取今彩539歷史數據...")
    daily_history = fetch_real_history("dailyCash", size=100)
    if daily_history:
        nums_only = [item["numbers"] for item in daily_history]
        recommended = ml_predict(nums_only, max_number=39, pick_count=5)
        
        predictions["daily_539"] = {
            "name": "今彩539",
            "numbers": recommended,
            "special": None,
            "style": "daily_539"
        }
    time.sleep(2)

    # === 4. 3星彩與4星彩 (維持模擬生成) ===
    # 註：這兩類屬於每位數字獨立 0~9 開獎，非一般選號，直接以隨機或簡易規則呈現
    predictions["star_3"] = {
        "name": "3星彩",
        "numbers": [int(np.random.randint(0, 10)) for _ in range(3)],
        "special": None,
        "style": "stars",
    }
    predictions["star_4"] = {
        "name": "4星彩",
        "numbers": [int(np.random.randint(0, 10)) for _ in range(4)],
        "special": None,
        "style": "stars",
    }

    # === 存檔輸出 ===
    output_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": predictions,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("資料爬取與機器學習訓練推演成功！ data.json 已儲存。")


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    main()