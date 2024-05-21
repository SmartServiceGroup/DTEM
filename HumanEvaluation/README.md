## 数据格式

四个任务下都有一些推荐结果 {模型名}.json。
这些文件是从实验结果里拷贝过来的。
这些文件的格式是相同的：
[
    {
        "ID":[
            [Hit GTs],          <-- 这里不需要这个信息
            [ID recommended]
        ]
    }
]

ID是节点在图中的编号，这个不同任务有所不同，具体情况是：
开发者ID来自 GNN/DataPreprocess/full_graph/content/contributors.json
仓库ID来自 GNN/DataPreprocess/full_graph/content/repositories.json
PR ID来自 GNN/DataPreprocess/full_graph/content/prs.json
