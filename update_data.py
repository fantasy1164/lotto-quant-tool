import json
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def create_features_and_labels(history_numbers, max_number=49, lookback=5):
    """
    特徵工程：將歷史號碼轉換為機器學習矩陣
    history_numbers: 過去每期的中獎號碼列表，例如 [[3, 12, ...], [5, 18, ...]]，越新的在越前面
    lookback: 參考過去幾期的資料來預測下一期
    """
    X = []
    y = []

    # 為了方便計算，將歷史資料轉為 0 和 1 的矩陣 (期數 x 號碼)
    # 例如：如果當期有 5 號，則該期第 5 個位置為 1，其餘為 0
    total_periods = len(history_numbers)
    matrix = np.zeros((total_periods, max_number + 1))
    for i, nums in enumerate(history_numbers):
        for num in nums:
            if 1 <= num <= max_number:
                matrix[i, num] = 1

    # 建立訓練樣本 (從最舊的資料開始往新資料滾動)
    # 每一號碼獨立作為一個樣本
    for t in range(total_periods - lookback - 1, -1, -1):
        # t 是我們要預測的目標期數 (t=0 代表最新一期)
        # 我們用 t+1, t+2, ..., t+lookback 的資料作為特徵
        for num in range(1, max_number + 1):
            # 特徵：該號碼在過去幾期的出現狀況 (1 或 0)
            feature = matrix[t + 1 : t + 1 + lookback, num]
            # 額外特徵：過去這幾期內總共出現了幾次 (冷熱門度)
            freq = np.sum(feature)
            final_feature = np.append(feature, freq)

            X.append(final_feature)
            y.append(matrix[t, num])  # 目標：第 t 期該號碼是否有開出

    return np.array(X), np.array(y), matrix


def ml_predict(history_numbers, max_number, pick_count):
    """
    使用隨機森林訓練並預測下一期號碼
    """
    lookback = 5
    # 如果歷史資料太少，無法訓練，則回傳隨機號碼
    if len(history_numbers) < (lookback + 2):
        import random

        return sorted(random.sample(range(1, max_number + 1), pick_count))

    # 1. 建立特徵與標籤
    X, y, matrix = create_features_and_labels(
        history_numbers, max_number, lookback
    )

    # 2. 初始化並訓練隨機森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 3. 準備「最新一期」的特徵，用來預測下一期 (即 t=-1 的未來)
    next_features = []
    for num in range(1, max_number + 1):
        # 拿最新的 lookback 期作為輸入特徵
        feature = matrix[0:lookback, num]
        freq = np.sum(feature)
        final_feature = np.append(feature, freq)
        next_features.append(final_feature)

    # 4. 預測開出機率 (predict_proba 會回傳 [不開出機率, 開出機率])
    probabilities = model.predict_proba(np.array(next_features))[:, 1]

    # 5. 結合號碼與機率，依機率由大到小排序
    num_prob_pairs = list(zip(range(1, max_number + 1), probabilities))
    num_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    # 6. 取出前 N 個機率最高的號碼並排序
    recommended_numbers = [pair[0] for pair in num_prob_pairs[:pick_count]]
    return sorted(recommended_numbers)


def main():
    # 模擬從爬蟲抓下來的歷史中獎號碼 (實際開發時，請用爬蟲抓取真實歷史數據)
    # 這裡以大樂透（49選6）和今彩539（39選5）為範例，隨機塞一些假歷史數據
    np.random.seed(42)
    dummy_lotto_history = [
        list(np.random.choice(range(1, 50), 6, replace=False))
        for _ in range(50)
    ]
    dummy_539_history = [
        list(np.random.choice(range(1, 40), 5, replace=False))
        for _ in range(50)
    ]

    predictions = {}

    # 1. 大樂透預測 (49選6)
    lotto_nums = ml_predict(dummy_lotto_history, max_number=49, pick_count=6)
    predictions["lotto"] = {
        "name": "大樂透",
        "numbers": lotto_nums,
        "special": int(np.random.randint(1, 50)),  # 特別號先以隨機代替
        "style": "lotto",
    }

    # 2. 今彩539預測 (39選5)
    daily_539_nums = ml_predict(
        dummy_539_history, max_number=39, pick_count=5
    )
    predictions["daily_539"] = {
        "name": "今彩539",
        "numbers": daily_539_nums,
        "special": None,
        "style": "daily_539",
    }

    # 3. 威力彩預測 (第一區 38選6)
    dummy_super_history = [
        list(np.random.choice(range(1, 39), 6, replace=False))
        for _ in range(50)
    ]
    super_nums = ml_predict(dummy_super_history, max_number=38, pick_count=6)
    predictions["super_lotto"] = {
        "name": "威力彩",
        "numbers": super_nums,
        "special": int(np.random.randint(1, 9)),
        "style": "super_lotto",
    }

    # 4. 3星彩與4星彩 (因為是 0~9 獨立數位，可用獨立隨機或個別訓練，此處先簡化處理)
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

    # 打包 JSON
    output_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": predictions,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("機器學習模型推演成功，data.json 已更新！")


if __name__ == "__main__":
    main()