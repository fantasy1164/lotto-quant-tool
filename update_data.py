import json
import random
from datetime import datetime


def fetch_lottery_data():
    """
    這裡放你的爬蟲邏輯 (例如使用 requests + BeautifulSoup)
    目前先用模擬的歷史數據代替
    """
    # 範例：假設這是你爬回來的最新大樂透歷史資料
    history_data = [
        {"period": "113000002", "numbers": [3, 12, 24, 33, 41, 45], "special": 8},
        {"period": "113000001", "numbers": [5, 18, 22, 30, 42, 49], "special": 15},
    ]
    return history_data


def train_and_predict(history):
    """
    這裡放你的機器學習模型推演邏輯 (例如冷熱門統計、LSTM 或隨機森林)
    目前先用隨機推薦作為 Placeholder
    """
    # 這裡可以讀取歷史數據進行特徵工程與模型預測
    predicted_numbers = sorted(random.sample(range(1, 50), 6))
    predicted_special = random.randint(1, 49)

    return {"numbers": list(predicted_numbers), "special": predicted_special}


def main():
    # 1. 爬取資料
    history = fetch_lottery_data()

    # 2. 模型預測
    prediction = train_and_predict(history)

    # 3. 打包成前端需要的 JSON 格式
    output_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history": history,
        "prediction": prediction,
    }

    # 4. 寫入檔案
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("data.json 更新成功！")


if __name__ == "__main__":
    main()