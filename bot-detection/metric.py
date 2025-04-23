import json, os

if os.path.exists("./data/bots.json"):
    with open("./data/bots.json", "r") as f:
        bots = json.load(f)
else:
    with open("./data/user.json", "r") as f:
        users = json.load(f)
        
    bots = []

    for i, user in enumerate(users):
        if users[user]["label"] == "bot":
            bots.append([i, user])

    json.dump(bots, open("./data/bots.json", "w"), indent=4)


bot_index = [x[0] for x in bots]

def BIN_metric():
    with open("./data/BIN_result.json") as f:
        bin_result = json.load(f)
    
    predicted_index = []
    for i, result in enumerate(bin_result):
        if result == 1:
            predicted_index.append(i)
    
    true_positive = len(set(predicted_index) & set(bot_index))
    false_positive = len(set(predicted_index)) - true_positive
    false_negative = len(set(bot_index)) - true_positive
    true_negative = 10000 - true_positive - false_positive - false_negative
    
    accuracy = (true_positive + true_negative) / 10000
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    
    print("BIN Metric")
    print(true_positive, false_positive, false_negative, true_negative)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    
def BIMAN_metric():
    with open("./data/BIMAN_result.json") as f:
        bin_result = json.load(f)
    
    predicted_index = []
    for i, result in enumerate(bin_result):
        if result < 0.5:
            predicted_index.append(i)
            
    true_positive = len(set(predicted_index) & set(bot_index))
    false_positive = len(set(predicted_index)) - true_positive
    false_negative = len(set(bot_index)) - true_positive
    true_negative = 10000 - true_positive - false_positive - false_negative
    
    accuracy = (true_positive + true_negative) / 10000
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    
    print("BIMAN Metric")
    print(true_positive, false_positive, false_negative, true_negative)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)


BIN_metric()
BIMAN_metric()