说明这个实验的数据是怎么获得的：
假设PRReviewwer为根目录，数据包括 
data/五个数据文件 和 metric/data/一个数据文件

data目录下的文件是 /collect_data.py 和 split_data.py 产出的
collect_data.py 生成了 pr_reviewer.json 和 pr_reviewers.json
split_data.py 将 pr_reviewer.json 按照 8：1：1 分割成 test/train/valid.json三个文件，所以内容格式是一致的。

#### pr_reviewer.json:
四元组序列
- 第一个元素：一个pr的repo_idx
- 第二个元素：一个pr的pr_idx
- 第三个元素：该pr的一个正样本reviewer，取自真实 reviewers 集合的第一个属于已知 contributor 集合的。一个 pr 只产生一个正样本。
- 第四个元素：随机出来的一个负样本reviewer
pr来自完整的pr数据，构成了简单NN网络监督训练的数据集，整个数据集有大约5w组正负样本对

#### pr_reviewers.json
pr的真实reviewer字典
- key的形式是 repo_idx # pr_idx
- value是真实reviewer 的 contributor_idx 序列

metric/data 下的文件是 metric/collect_data.py 产生的

#### metric/data/dataset_valid_test.json
四元组序列
- 第一个元素：一个pr的repo_idx
- 第二个元素：一个pr的pr_idx
- 第三个元素：search_scope，即需要从这个集合里去推荐出Top-K。
    与 repo_idx 号仓库存在 commit 关系的所有 contributor 组成了这个集合。
- 第四个元素：GT，该pr的所有真实reviewer，来自 data/pr_reviews.json

pr来自 data/test 和 data/valid 的并集。
实验时要求一个 pr 所在的仓库至少有10个开发者，即 search_scope 不短于10，否则丢弃。
过滤 search_scope 长度前有 10442 项，平均从 17 个item中推荐 1.6 个gt
过滤后长度有 4953 项，平均从 32 个item中推荐 1.8 个gt

#### metric/data/dataset_valid_test_modified.json
上述数据集中存在无效 gt 的问题，即部分 GT 在 search scope 中没有出现
这个数据集修复了这一问题

