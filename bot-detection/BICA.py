import json, os
import numpy as np

if os.path.exists("./data/bica.json"):
    with open("./data/bica.json", 'r') as file:
        bica_data = json.load(file)
else:
    with open("./data/user.json", 'r') as file:
        user_data = json.load(file)
        
    bica_data = {
        "Uniq.File.Exten": [],
        "Tot.FilesChanged": [],
        "Std.File.pCommit": [],
        "Tot.uniq.Projects": [],
        "Avg.File.pCommit": [],
        "Median.Project.pCommit": []
    }
    for username in user_data:
        current_user_data = {}
        
        files_per_commit = []
        filenames_changed = set()
        commits_per_project = []
        
        for repo in user_data[username]["commits"]:
            commits_per_project.append(len(user_data[username]["commits"][repo]))
            for commit in user_data[username]["commits"][repo]:
                files_per_commit.append(len(user_data[username]["commits"][repo][commit]["filenames"]))
                for filename in user_data[username]["commits"][repo][commit]["filenames"]:
                    filenames_changed.add(filename)
        
        total_files_changed = sum(files_per_commit)
        unique_files_extension = len(filenames_changed)
        std_file_pCommit = 0 if total_files_changed == 0 else np.std(files_per_commit)
        avg_file_pCommit = 0 if total_files_changed == 0 else np.mean(files_per_commit)
        total_unique_projects = len(user_data[username]["commits"])
        median_commit_pProject = 0 if len(commits_per_project) == 0 else np.median(commits_per_project)
        
        bica_data["Uniq.File.Exten"].append(total_files_changed)
        bica_data["Tot.FilesChanged"].append(unique_files_extension)
        bica_data["Std.File.pCommit"].append(std_file_pCommit)
        bica_data["Tot.uniq.Projects"].append(total_unique_projects)
        bica_data["Avg.File.pCommit"].append(avg_file_pCommit)
        bica_data["Median.Project.pCommit"].append(median_commit_pProject)    
        
        
    with open("./data/bica.json", 'w') as file:
        json.dump(bica_data, file)
    

from pypmml import Model
import pandas as pd

model = Model.fromFile("./data/BICA_model.pmml")

bica_data = [{k: bica_data[k][i] for k in model.inputNames} for i in range(len(bica_data["Uniq.File.Exten"]))]
input_data = pd.DataFrame(bica_data).astype("double")  
predictions = model.predict(input_data)

result = [float(x) for x in predictions["probability(0)"]]

with open("./data/BICA_result.json", 'w') as file:
    json.dump(result, file)
