import json

with open("./data/user.json", 'r') as file:
    user_data = json.load(file)

print("Total length:", len(user_data))
result = []
for username in user_data:
    robot = 0
    username = "%s <%s>" % (username, user_data[username]['email'])
    matching_patterns = [
        "-bot",
        "bot-",
        "-Bot",
        "Bot-",
        "[bot",
        "bot]",
        "bot ",
        "Bot "
    ]
    for pattern in matching_patterns:
        if pattern in username:
            robot = 1
    
    result.append(robot)
    
    
        
print("Bot count:", sum(result))
with open("./data/BIN_result.json", 'w') as file:
    json.dump(result, file)