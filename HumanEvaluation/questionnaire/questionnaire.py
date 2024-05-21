import sys, json

# python quesionnaire.py <data per questionnaire>

def make_selection_quesiton(title, description, options) -> str:
    text = title + "[单选题]"
    if description:
        text += "({})".format(description)
    text += '\n'
    text += '\n'.join(options)
    text += '\n\n'
    return text

def make_multi_selection_question(title, description, options) -> str:
    text = title + "[多选题]"
    if description:
        text += "({})".format(description)
    text += '\n'
    text += '\n'.join(options)
    text += '\n\n'
    return text

def make_matrix_selection_question(title, description, questions, options) -> str:
    text = title + "[矩阵单选题]"
    if description:
        text += "({})".format(description)
    text += '\n'
    text += ' '.join(options)
    text += '\n'
    text += '\n'.join(questions)
    text += '\n\n'
    return text


assert len(sys.argv) == 2
data_per_questionnaire = int(sys.argv[1])

questionnaire_id = 0
data_index = 0
data_total_num = 30

while data_index + data_per_questionnaire <= data_total_num:
    dst_file = './result/questionnaire_{}.txt'.format(questionnaire_id)
    
    data_required = []
    for data_iter_index in range(data_index, data_index + data_per_questionnaire):
        src_file = './data/{}.json'.format(data_iter_index)
        with open(src_file, 'r') as f:
            data_iter = json.load(f)
            data_required.append(data_iter)
    
    with open(dst_file, 'w') as f:
        
        text = '\n关于Github推荐服务测评的问卷\n'
        f.write(text)
        text = ""

        # ---------------------- 用户开发经验调研部分 ------------------------
        title = "您的开发经验时长"
        description = ""
        options = ["2年以内", "2-5年", "5-10年", "10年以上"]
        text += make_selection_quesiton(title, description, options)

        title = "您使用Github的时长"
        options = ["从未使用", "2年以内", "5-10年", "10年以上"]
        text += make_selection_quesiton(title, description, options)
        
        title = "您是否在Github上对开源项目做出贡献"
        options = ["从未贡献", 
                   "曾提出过 Pull Request, 但并未被采纳", 
                   "曾提出过 Pull Request, 而且被项目管理者采纳",
                   "曾提出过 Issue, 但并未被采纳",
                   "曾提出过 Issue, 而且被项目管理者采纳",
                   "曾提出过 Issue, 并提出相应的 Pull Request",
                   "曾参与过开源项目的管理维护工作"
                   ]
        text += make_multi_selection_question(title, description, options)
        
        text += "===分页===\n\n"
        f.write(text)
    
        # ---------------------- ContributionRepo ------------------
        text = ""
        first_title = "下面的每道题中会分别给出一个Github上的开发者和一个Github上的开源仓库，以及它们的Github链接。我们感兴趣的是，你认为在多大程度上，该开发者适合对题目给出的仓库做出贡献。请你浏览该用户和该仓库的基本信息，并作出你的评价\\n"
        first_question = True
        
        for data in data_required:
            questions = data['questions']
            for question in questions:
                if not question[4] == "ContributionRepo":
                    continue
                contributor, repo = question[1], question[3]
                title = "用户：[{}](https://github.com/{}), 开源仓库：[{}](https://github.com/{})".format(contributor, contributor, repo, repo)            
                description = ""
                options = ['1', '2', '3', '4', '5']
                questions = [
                    "在多大程度上，该用户具有该项目所需的技术能力?",
                    "在多大程度上，该用户具有与该项目相似的开发经验?"
                ]
                if first_question:
                    title = first_title + title
                    questions[0] += "(1表示最少,5表示最多,下同)"
                    first_question = False

                
                text += make_matrix_selection_question(title, description, questions, options)
        
        text += "===分页===\n\n"
        f.write(text)
        
        # ---------------------- PRReviewer ------------------------
        text = ""
        first_title = "下面的每道题中会分别给出Github上的一个 PR(Pull Request) 和一个开发者，以及它们的链接。我们感兴趣的是，你认为在多大程度上，该开发者适合对作为 Reviewer 来审查该项 PR。请你浏览给出的 Pull Request和开发者的基本信息，并作出你的评价\\n"
        first_question = True
        
        for data in data_required:
            questions = data['questions']
            for question in questions:
                if not question[4] == "PRReviewer":
                    continue
                
                pr, contributor = question[1], question[3]
                pr_url = pr.replace("##", "/pull/")
                title = "Pull Request：[{}](https://github.com/{}), 开发者：[{}](http://github.com/{})".format(pr, pr_url, contributor, contributor)            
                if first_question:
                    title = first_title + title
                    first_question = False
                description = ""
                options = ['1', '2', '3', '4', '5']
                questions = [
                    "在多大程度上，该用户精通该 PR 相关的技术，并能够识别 PR 中的潜在问题?",
                ]
                
                text += make_matrix_selection_question(title, description, questions, options)
        
        text += "===分页===\n\n"
        f.write(text)

        # ---------------------- RepoMaintainer --------------------
        text = ""
        first_title = "下面的每道题中会分别给出Github上的一个开源仓库和一个开发者，以及它们的链接。我们感兴趣的是，你认为在多大程度上，该开发者适合参与该项目的管理和维护工作。请你浏览给出的仓库和开发者的基本信息，并作出你的评价\\n"
        first_question = True
        
        for data in data_required:
            questions = data['questions']
            for question in questions:
                if not question[4] == "RepoMaintainer":
                    continue
                
                repo, contributor = question[1], question[3]
                title = "开源仓库：[{}](https://github.com/{}), 用户：[{}](http://github.com/{})".format(repo, repo, contributor, contributor)            
                if first_question:
                    title = first_title + title
                    first_question = False
                description = ""
                options = ['1', '2', '3', '4', '5']
                questions = [
                    "在多大程度上，该用户能够较好地完成该项目的运行维护工作？",
                    "在多大程度上，该用户的技术栈能够涵盖项目所设计的技术？"
                ]
                
                text += make_matrix_selection_question(title, description, questions, options)
        
        text += "===分页===\n\n"
        f.write(text)

        # ---------------------- SimDeveloper ----------------------
        text = ""
        first_title = "下面的每道题中会分别给出两个Github上的两个用户，以及它们的链接。我们感兴趣的是，你认为在多大程度上，这两个用户具有相似的技术特征。请你浏览给出的开发者的基本信息，并作出你的评价\\n"
        first_question = True
        
        for data in data_required:
            questions = data['questions']
            for question in questions:
                if not question[4] == "SimDeveloper":
                    continue
                
                contributor1, contributor2 = question[1], question[3]
                title = "用户1：[{}](https://github.com/{}), 用户2：[{}](http://github.com/{})".format(contributor1, contributor1, contributor2, contributor2)            
                if first_question:
                    title = first_title + title
                    first_question = False
                description = ""
                options = ['1', '2', '3', '4', '5']
                questions = [
                    "在多大程度上，这两个用户具有相似的技术栈？",
                    "在多大程度上，这两个用户具有相似的开发工作经验？"
                ]
                
                text += make_matrix_selection_question(title, description, questions, options)
        
        f.write(text)
    data_index += data_per_questionnaire
    questionnaire_id += 1
