说明这个实验的数据是怎么获得的：
假设 RepoMaintainer 为根目录，数据包括 
data/四个数据文件 和 metric/data/一个数据文件

data目录下的文件是 /collect_data.py 和 split_data.py 产出的
collect_data.py 生成了 repo_organizer_samples.json
split_data.py 将 repo_organizer_samples.json 按照 8：1：1 分割成 test/train/valid.json三个文件，所以内容格式是一致的。

#### repo_organizer_samples.json:
三元组序列
- 第一个元素：一个repo的repo_idx
- 第二个元素：该 repo 的一个正样本，即适合的维护者
- 第三个元素：该 repo 的一个负样本

产生的方式是，遍历全部 repo，如果存在其 owner 的信息，即这仓库所有者也在数据集中，就将 owner 作为正样本，并随机选取一个负样本。
这样大约产生 2w 个正负样本对，意味着大约 2w 个仓库参与这一任务。
这一数据构成简单NN网络的监督训练数据。


#### metric/data/dataset_valid_test.json:
三元组序列
- 第一个元素：一个 repo 的repo_idx
- 第二个元素：search_scope，即需要从这个集合里去推荐出Top-K。
    对于 repo，其 search_scope 定义成所有贡献过该仓库的开发者。
- 第三个元素：GT，构建方法类似于 ContributionRepo 任务的做法，对一个 repo 将 test/valid 中的正样本收集起来，当作 GT

repo 来自于 test/valid 中的所有 repo 项，进行实验时过滤掉了 search_scope 短于 10 的样本
过滤 search_scope 长度前有 4938 项，平均从 1.38 个 item 中推荐 1 个 gt
过滤后长度有 35 项，平均从 17 个item中推荐 1 个 gt


