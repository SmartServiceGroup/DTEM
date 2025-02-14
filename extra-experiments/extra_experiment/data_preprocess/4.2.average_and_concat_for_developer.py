
import os, sys, json, torch, dgl, tqdm
import numpy as np

from dgl import load_graphs
from dgl.data.utils import save_graphs

from utils import RepositoryFeatureLoader, IssueFeatureLoader, PRFeatureLoader

structure_graph_file            = "./full_graph/structure_graph.bin"
issue_idx_file                  = "./full_graph/content/issues.json"
pr_idx_file                     = "./full_graph/content/prs.json"
repo_idx_file                   = "./full_graph/content/repositories.json"
feature_dir                     = "../../NodeFeatureInitializer/export"
metapath_node_embedding_file    = "./cache/full_graph/node_metapath_embedding.bin"
# dst_graph_file                  = "./full_graph/structure_graph_with_average_feature_with_metapath.bin"
dst_graph_file                  = "./full_graph/structure_graph_with_average_feature_without_metapath.bin"



if __name__ == "__main__":
    device = torch.device("cpu")
    issue_feature_loader = IssueFeatureLoader(issue_idx_file)
    repo_feature_loader = RepositoryFeatureLoader(repo_idx_file)
    pr_feature_loader = PRFeatureLoader(pr_idx_file)

    repo_text_feature = repo_feature_loader.load_text_feature_for_repo_node(repo_embedding_file=os.path.join(feature_dir,"repo_text_embedding.pkl"), embed_size=768, device=torch.device("cpu"))
    repo_code_feature = repo_feature_loader.load_code_feature_for_repo_node(repo_embedding_file=os.path.join(feature_dir,"repo_code_embedding.pkl"), embed_size=768, device=torch.device("cpu"))
    repo_language_feature = repo_feature_loader.load_language_feature_for_repo_node(repo_embedding_file=os.path.join(feature_dir,"repo_languages_pca.pkl"), embed_size=256, device=torch.device("cpu"))
    repo_topic_feature = repo_feature_loader.load_topic_feature_for_repo_node(repo_embedding_file=os.path.join(feature_dir,"repo_topics_pca.pkl"), embed_size=256, device=torch.device("cpu"))
    repo_feature = torch.cat([repo_text_feature, repo_code_feature, repo_language_feature, repo_topic_feature], dim=1)
    
    pr_text_feature = pr_feature_loader.load_text_feature_for_pr_node(os.path.join(feature_dir,"pr_text_embedding.pkl"), embed_size=768, device=torch.device("cpu"))
    pr_code_feature = pr_feature_loader.load_code_feature_for_pr_node(os.path.join(feature_dir,"pr_code_embedding.pkl"), embed_size=768, device=torch.device("cpu"))
    pr_feature = torch.cat([pr_text_feature, pr_code_feature], dim=1)
    
    issue_feature = issue_feature_loader.load_embedding_for_issue_node(
        text_embedding_file=os.path.join(feature_dir, "issue_text_embedding.pkl"), embed_size=768,
        device=device,
        load_cache=False
    )
    
    hg = load_graphs(structure_graph_file)[0][0]
    print(hg.etypes)
    contributor_num = hg.num_nodes("contributor")
    
    contributor_feature = np.zeros((contributor_num, 2048), dtype=np.float32)
    for x in tqdm.tqdm(range(contributor_num)):
        watch_repos = hg.out_edges(x, "uv", "contributor_watch_repo")
        star_repos = hg.out_edges(x, "uv", "contributor_star_repo")
        commit_repos = hg.in_edges(x, "uv", "repo_committed_by_contributor")
        related_repos = torch.concat((watch_repos[1], star_repos[1], commit_repos[0]))
        
        related_issues = hg.out_edges(x, "uv", "contributor_propose_issue")[1]
        related_prs = hg.out_edges(x, "uv", "contributor_propose_pr")[1]
        
        text_feature_for_contributor = []
        code_feature_for_contributor = []
        language_feature_for_contributor = []
        topic_feature_for_contributor = []
        
        for repo in related_repos:
            text_feature_for_contributor.append(repo_text_feature[repo])
            code_feature_for_contributor.append(repo_code_feature[repo])
            topic_feature_for_contributor.append(repo_topic_feature[repo])
            language_feature_for_contributor.append(repo_language_feature[repo])
        for issue in related_issues:
            text_feature_for_contributor.append(issue_feature[issue])
        for pr in related_prs:
            text_feature_for_contributor.append(pr_text_feature[pr])
            code_feature_for_contributor.append(pr_code_feature[pr])
        
        if len(text_feature_for_contributor) > 0:
            text_feature_for_contributor = torch.stack(text_feature_for_contributor).mean(dim=0)
        else:
            text_feature_for_contributor = torch.zeros([768], dtype=torch.float)
        
        if len(code_feature_for_contributor) > 0:
            code_feature_for_contributor = torch.stack(code_feature_for_contributor).mean(dim=0)
        else:
            code_feature_for_contributor = torch.zeros([768], dtype=torch.float)
        
        if len(language_feature_for_contributor) > 0:
            language_feature_for_contributor = torch.stack(language_feature_for_contributor).mean(dim=0)
        else:
            language_feature_for_contributor = torch.zeros([256], dtype=torch.float)
            
        if len(topic_feature_for_contributor) > 0:
            topic_feature_for_contributor = torch.stack(topic_feature_for_contributor).mean(dim=0)
        else:
            topic_feature_for_contributor = torch.zeros([256], dtype=torch.float)
        
        feature_for_developer = torch.cat((text_feature_for_contributor, code_feature_for_contributor, language_feature_for_contributor, topic_feature_for_contributor), dim=0)
        contributor_feature[x] = feature_for_developer
    
    contributor_feature = torch.FloatTensor(contributor_feature.tolist()).to(device)
    
    metapath_node_embedding = torch.load(metapath_node_embedding_file, map_location=device)

    # hg.nodes["pr"].data["feat"] = torch.cat([metapath_node_embedding["pr"], pr_feature], dim=1)
    # hg.nodes["repository"].data["feat"] = torch.cat([metapath_node_embedding["repository"], repo_feature], dim=1)
    # hg.nodes["issue"].data["feat"] = torch.cat([metapath_node_embedding["issue"], issue_feature], dim=1)
    # hg.nodes["contributor"].data["feat"] = torch.cat([metapath_node_embedding["contributor"], contributor_feature], dim=1)
    hg.nodes["pr"].data["feat"] = pr_feature
    hg.nodes["repository"].data["feat"] =  repo_feature
    hg.nodes["issue"].data["feat"] = issue_feature
    hg.nodes["contributor"].data["feat"] = contributor_feature

    save_graphs(dst_graph_file, [hg])



"""

To use the saved graph, try this: 

```python
from dgl import load_graphs

dst_graph_file = "./full_graph/structure_graph_with_average_feature_without_metapath.bin"
# dst_graph_file = "./full_graph/structure_graph_with_average_feature_with_metapath.bin"
graph = load_graphs(dst_graph_file)[0][0]

print((
    graph.nodes['pr'].data['feat'].shape, 
    graph.nodes['repository'].data['feat'].shape, 
    graph.nodes['issue'].data['feat'].shape, 
    graph.nodes['contributor'].data['feat'].shape
))
```

|-------------|--------|-------------------|----------------------|
| entity      | count  | dim-with-metapath | dim-without-metapath |
|-------------|--------|-------------------|----------------------|
| pr          | 379496 | 1792              | 1536                 |
| repository  | 50000  | 2304              | 2048                 |
| issue       | 692554 | 1024              | 768                  |
| contributor | 394474 | 2304              | 2048                 | 
|-------------|--------|-------------------|----------------------|
"""
