反正这个工作之后主要是我做了, 那我就瞎说了. 反正也没别人看. 

这个目录下有好几个数据文件. 解释一下. 假设当前目录为 ContributionRepo. 


### data/ 

这个目录下的文件都是拿来训练模型用的.

#### data/user_watch_repos.json: 
由 collect_data.py 获得. 
三元组序列
- 第一个元素：contributor_idx
- 第二个元素：contributor_idx 的一个正样本，（指适合推荐给他）
- 第三个元素：contributor_idx 的一个负样本

遍历了全部开发者，对一个开发者，他的正样本定义成曾经 commit/watch 过的仓库。
全部这样的仓库都加入了正样本，并随机选取了等数量负样本。
形成了约 3w 个正负样本对，构成简单NN网络的监督训练数据。


#### data/{test,valid,train}.json

就是将上面的文件分割出来, 得到的三个临时文件. 也是用来训练数据的. 


### metric

#### metric/data/dataset_valid_test.json

这个文件由 metric/collect_data.py 取得. 
三元组序列
- 第一个分量是某个贡献者的 id; 
- 第二个分量是 search_scope，要从这个集合中推荐 Top-K
    search_scope 的定义方式是，该开发者 commit/star/watch 过的所有仓库，
    以及 following/followed by 该开发者的开发者所 commit 过的所有仓库。
- 第三个分量是 GT，开发者 commit/watch 过的仓库。
    这是将 test 和 valid 的正样本收集了起来，因此不完整，也可能不止一个

整个数据是从 data/test.json 和 data/valid.json 的并得到的
实验时过滤掉 GT 短于 5 的样本，即 test valid 中有少于 5 个该开发者的正样本
筛选前有 5583 个开发者，平均从 47 个 item 中推荐 1.7 个 gt
筛选后有 42 个开发者，平均从 217 个 item 中推荐 8.4 个 gt

