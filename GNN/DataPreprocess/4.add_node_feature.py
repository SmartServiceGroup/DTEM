import os
import sys
import json

import torch
import dgl

from dgl import load_graphs
from dgl.data.utils import save_graphs

from utils import RepositoryFeatureLoader, IssueFeatureLoader, PRFeatureLoader

"""
    上一步得到的文件是 ./cache/full_graph/node_metapath_embedding.bin, 
    包含了经过 metapath 训练得到的节点 embedding.
    我们将这些embedding与节点的其他特征拼接起来, 得到最终的节点特征.

    Note: 
        没有训练: 这一步只是拼接.
        输出: ./full_graph/structure_graph_with_node_feature_with_metapath.bin
"""


structure_graph_file            = "./full_graph/structure_graph.bin"
issue_idx_file                  = "./full_graph/content/issues.json"
pr_idx_file                     = "./full_graph/content/prs.json"
repo_idx_file                   = "./full_graph/content/repositories.json"
feature_dir                     = "../../NodeFeatureInitializer/export"
metapath_node_embedding_file    = "./cache/full_graph/node_metapath_embedding.bin"
dst_graph_file                  = "./full_graph/structure_graph_with_node_feature_with_metapath.bin"

if __name__ == "__main__":
    device = torch.device("cpu")
    issue_feature_loader = IssueFeatureLoader(issue_idx_file)
    repo_feature_loader = RepositoryFeatureLoader(repo_idx_file)
    pr_feature_loader = PRFeatureLoader(pr_idx_file)

    pr_feature = pr_feature_loader.load_embedding_for_pr_node(
        pr_text_embdding_file=os.path.join(feature_dir, "pr_text_embedding.pkl"), text_embed_size=768,
        pr_code_embedding_file=os.path.join(feature_dir, "pr_code_embedding.pkl"), code_embed_size=768,
        device=device,
        load_cache=False
    )
    repo_feature = repo_feature_loader.load_embedding_for_repo_node(
        repo_code_embedding_file=os.path.join(feature_dir, "repo_code_embedding.pkl"), code_embed_size=768,
        repo_text_embedding_file=os.path.join(feature_dir, "repo_text_embedding.pkl"), text_embed_size=768,
        repo_language_embedding_file=os.path.join(feature_dir, "repo_languages_pca.pkl"), language_embed_size=256,
        repo_topic_embedding_file=os.path.join(feature_dir, "repo_topics_pca.pkl"), topic_embed_size=256,
        device=device,
        include_topic=True,
        load_cache=False
    )
    issue_feature = issue_feature_loader.load_embedding_for_issue_node(
        text_embedding_file=os.path.join(feature_dir, "issue_text_embedding.pkl"), embed_size=768,
        device=device,
        load_cache=False
    )

    hg = load_graphs(structure_graph_file)[0][0]
    # add reverse edges for directed graphs
    # hg = dgl.transforms.AddReverse(sym_new_etype=True)(hg)
    metapath_node_embedding = torch.load(metapath_node_embedding_file, map_location=device)

    # 这里将 pr, repo, issue, contributor 的 embedding 与 metapath 的 embedding 拼接. 
    # TODO 我们可以考虑不用 pric 的 embedding, 将它们随机化. 再做后面的训练
    # 然后比较各个下游任务上的效果. 
    hg.nodes["pr"].data["feat"] = torch.cat([metapath_node_embedding["pr"], pr_feature], dim=1)
    hg.nodes["repository"].data["feat"] = torch.cat([metapath_node_embedding["repository"], repo_feature], dim=1)
    hg.nodes["issue"].data["feat"] = torch.cat([metapath_node_embedding["issue"], issue_feature], dim=1)
    hg.nodes["contributor"].data["feat"] = metapath_node_embedding["contributor"]

    # edge_idx = 8
    # for etype in hg.etypes:
    #     if "rev" in etype:
    #         hg.edges[etype].data["reltype"] = hg.edges[etype[4:]].data["reltype"] + 8
    #         edge_idx += 1

    save_graphs(dst_graph_file, [hg])
    