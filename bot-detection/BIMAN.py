import json, os
import numpy as np

if os.path.exists("./data/biman.json"):
    with open("./data/biman.json", 'r') as file:
        biman_data = json.load(file)
else:
    with open("./data/BIN_result.json", 'r') as file:
        bin_result = json.load(file)
    
    with open("./data/BICA_result.json", 'r') as file:
        bica_result = json.load(file)
        
    with open("./data/BIM_result.json", 'r') as file:
        bim_result = json.load(file)
        
    biman_data = {
        "p": [],            # BICA
        "ratio": [],        # BIM
        "name": []          # BIN
    }
    
    for i in range(len(bin_result)):
        biman_data["p"].append(bica_result[i])        
        biman_data["ratio"].append(float(bim_result[i].split(";")[3]))
        biman_data["name"].append(bin_result[i])
        
    with open("./data/biman.json", 'w') as file:
        json.dump(biman_data, file)    

from pypmml import Model
import pandas as pd

model = Model.fromFile("./data/ensemble_model.pmml")

input_data = [{k: biman_data[k][i] for k in model.inputNames} for i in range(len(biman_data["p"]))]
input_data = pd.DataFrame(input_data).astype("double")  
predictions = model.predict(input_data)

result = [float(x) for x in predictions["probability(0)"]]

with open("./data/biman_result.json", 'w') as file:
    json.dump(result, file)

total = 0
for i, result_p in enumerate(result):
    if result_p < 0.5:
        print(biman_data["p"][i], biman_data["ratio"][i], biman_data["name"][i], result_p)
        total += 1
print(total)
