## 对比实验 1. 和 SOTA 的技能向量的对比. 

在这篇论文中: 

> Dev2vec: Representing domain expertise of developers in an embedding space

提到了另一种为开发者建模的方式, 也是给开发者搞一个嵌入向量, 

不过, 用的方法是, 将开发者的 Issue, Repo, API 三部分分别构建向量, 
然后拼接到一起. 

幸运的是, 这个论文开源了它的模型, 我们可以直接使用...话是这么说, 但数据还是要和它的对齐. 

这个 alpha 做的就是这个实验. 我们希望证明, 用他的方法在 SimDeveloper 上的效果, 不如用
我们的 GCN 的方法来得好.

实验结果如下. 

| Model          | Precision | Recall | F1    |
|----------------|-----------|--------|-------|
| Our Model      | 0.920     | 0.983  | 0.950 | 
| Compared Model | 0.779     | 0.902  | 0.836 |
| Topic Model    | 0.943     | 0.767  | 0.846 |

| Model          | HR@1  | HR@3  | HR@5  | HR@10 | MRR   |
|----------------|-------|-------|-------|-------|-------|
| Our Model      | 0.434 | 0.705 | 0.801 | 0.886 | 0.591 |
| Compared Model | 0.385 | 0.655 | 0.769 | 0.868 | 0.547 |
| Topic Model    | 0.366 | 0.635 | 0.745 | 0.843 | 0.525 |


> `Our Model` 是我们论文中我们的方法; `Compared Model` 就是上述论中的方法; 
> 
> `Topic Model` 是学长在我们的论文中, 提到的一个对比的方法.

