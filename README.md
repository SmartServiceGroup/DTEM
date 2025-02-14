# Introduction
This repository is the code submitted for paper **Automatically Deriving Developers’ Technical Expertise from the GitHub Social Network**

# How to Run
Below shows the repository structure. The correspondence between the table results in the paper and the code can be roughly found through the comments in the directory trees.
```
.
├── Comparisons
│   └── experiments
│       ├── alpha       # run this to get the result compared with Dev2Vec, in Table 4 & 12, SimDeveloper/Dev2Vec Method
│       ├── beta        # run this to get performance without watch in Table 9
│       ├── gamma       # run this to get the result of GAT Method results in Table 4,5,12
│       └── theta       # run this to get the result of No Social Network & No Social Network (w/o mp) in Table 6
├── GHCrawler # run this to get the dataset in the research scope
│   └── export
├── GNN # run code here to get the detailed GNN performance in table 7 & table 8
│   ├── DataPreprocess
│   ├── HGT
│   │   └── pretrained
│   ├── HetGAT
│   │   ├── bin
│   │   └── pretrained
│   ├── HetGCN
│   │   └── pretrained
│   ├── HetSAGE
│   │   ├── bin
│   │   └── pretrained
│   ├── RGCN
│   │   ├── bin
│   │   └── pretrained
│   └── Visualize
├── NodeFeatureInitializer # run code here to get initial technical expertise embedding of Issue, PR, Repository nodes.
│   ├── IssueEmbedding
│   ├── PREmbedding
│   ├── RepositoryCodeEmbedding
│   ├── RepositoryEmbedding
│   ├── export
│   └── parser
│       ├── tree-sitter-c-sharp
│       ├── tree-sitter-go
│       ├── tree-sitter-java
│       ├── tree-sitter-javascript
│       ├── tree-sitter-php
│       ├── tree-sitter-python
│       └── tree-sitter-ruby
├── RecommendationTasks # run code here to get performance in 4 downstream recommendation tasks which is shown in table 4 & table 5
│   ├── ContributionRepo
│   │   └── metric
│   ├── ContributionRepo_CF
│   │   └── metric
│   ├── RepoMaintainer
│   │   └── metric
│   ├── RepoMaintainer_CF
│   │   └── metric
│   ├── SimDeveloper
│   │   └── metric
│   ├── SimDeveloper_CF
│   │   └── metric
│   └── TopicEmbedding # the baseline method in table 4 & table 5
├── T-Test # run code here to get the t-test result in table 10
├── extra-experiments
│   └── extra_experiment # codes here to run experiments on shrinked graph, shown in table 14 & 15
└── imgs
```

Recommended running configuration:

- OS: Ubuntu 20.04 
- CPU: Intel Xeon Gold 6254 @ 3.1GHz
- GPU: Nvidia A100 40GB
- mem: 125GB+
- SSD: 1TB
- python version: 3.7+ 
- pytorch version: 1.12.0+
- dgl version: 1.0.02+ 
