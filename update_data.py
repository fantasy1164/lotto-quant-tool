import json
import random
from datetime import datetime


def train_and_predict_all():
    """
    在這裡針對不同彩券的規則進行預測。
    未來你可以把 random.sample 替換成你訓練好的 AI 模型。
    """
    predictions = {}

    # 1. 大樂透 (Lotto 6/49): 1~49 選 6，特別號 1~49
    predictions["lotto"] = {
        "name": "大樂透",
        "numbers": sorted(random.sample(range(1, 50), 6)),
        "special": random.randint(1, 49),
        "style": "lotto",
    }

    # 2. 威力彩 (Super Lotto): 第一區 1~38 選 6，第二區 1~8 選 1
    predictions["super_lotto"] = {
        "name": "威力彩",
        "numbers": sorted(random.sample(range(1, 39), 6)),
        "special": random.randint(1, 8),
        "style": "super_lotto",
    }

    # 3. 今彩539 (Daily Cash 539): 1~39 選 5
    predictions["daily_539"] = {
        "name": "今彩539",
        "numbers": sorted(random.sample(range(1, 40), 5)),
        "special": None,
        "style": "daily_539",
    }

    # 4. 3星彩 (3-Star): 3 個 0~9 的獨立數字（位置不可調換）
    predictions["star_3"] = {
        "name": "3星彩",
        "numbers": [random.randint(0, 9) for _ in range(3)],
        "special": None,
        "style": "stars",
    }

    # 5. 4星彩 (4-Star): 4 個 0~9 的獨立數字
    predictions["star_4"] = {
        "name": "4星彩",
        "numbers": [random.randint(0, 9) for _ in range(4)],
        "special": None,
        "style": "stars",
    }

    return predictions


def main():
    # 這裡未來可以加入各類彩券的爬蟲，目前先模擬
    dummy_history = {
        "lotto": [
            {"period": "113000001", "numbers": [5, 18, 22, 30, 42, 49], "special": 15}
        ],
        "super_lotto": [
            {"period": "113000001", "numbers": [1, 10, 15, 23, 28, 35], "special": 4}
        ],
    }

    # 執行所有產品的預測
    predictions = train_and_predict_all()

    # 打包 JSON
    output_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history": dummy_history,
        "predictions": predictions,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("所有電腦型彩券資料更新成功！")


if __name__ == "__main__":
    main()