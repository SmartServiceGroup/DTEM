说明这个实验的数据是怎么获得的：
假设 SimDeveloper 为根目录，数据包括 
data/五个数据文件 和 metric/data/一个数据文件

data目录下的文件是 /collect_data.py 和 split_data.py 产出的
collect_data.py 生成了 org_user.json 和 sim_users.json
split_data.py 将 org_user.json 按照 8：1：1 分割成 test/train/valid.json三个文件，所以内容格式是一致的。

#### sim_user.json:
三元组序列
- 第一个元素：一个开发者的 contributor_idx
- 第二个元素：对该开发者的正样本
- 第三个元素：对该开发者的负样本，随机取得。

所有的开发者来自 GHCCrarler/cleaned/user_organization.txt，
这文件记录了 3w+ 个用户，每个用户记录了的全部所属组织。
收集出现所有的组织，并对每个组织（单人组织除外）中的每个成员生成一个正负样本对。正样本来自组织内，负样本在组织外随机。
共约 3.5w 个正负样本对，构成NN监督学习数据。

#### org_user.json
字典，每个组织拥有的用户信息
相当于将 GHCCrarler/cleaned/user_organization.txt 的映射关系逆转一下


#### metric/data/dataset_valid_test.json
三元组序列
- 第一个元素：一个 contributor 的 contributor_idx
- 第二个元素： search_scope，与该开发者贡献过同意仓库的所有开发者
- 第三个元素：GT，该开发者所属的全部组织包含的全部成员（甚至包括自己）

contributor 来自 data/test 和 data/valid 的并集。
实验时要求一个 contributor 至少有 5 个 gt，否则丢弃。
过滤 label 长度前有 4216 项，平均从 45 个 item 中推荐 230 个gt
过滤后长度有 3494 项，平均从 49 个 item 中推荐 278 个gt


#### metric/data/dataset_valid_test_modified.json
之前的数据中存在问题： GT 不去重，GT; search_scope 包含自身; 部分 GT 无效即不在 search scope 中

过滤 label 长度前有 2799 项，平均从 55 个 item 中推荐 9 个 GT
过滤后有 1303 项，平均从 86 个item 中推荐 17 个 gt


