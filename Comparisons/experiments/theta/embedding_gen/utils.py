
import numpy as np
from typing import *
import torch
from tqdm import tqdm

def kv_emb_to_tensor(
        emb_dict: Dict[str, np.ndarray], 
        idx: Dict[str, int], 
        device=torch.device('cpu')
) -> np.ndarray:
    '''
    将一个 k-v 格式的字典 (v为向量)
    根据 idx (k-i 格式, i: index), 
    转换为一个numpy矩阵, 
    将对应序号的嵌入向量填充进去.
    '''
    emb_size = next(iter(emb_dict.values())).shape[0]
    cnt = len(idx)

    ret = np.zeros((cnt, emb_size), dtype=np.float32)
    missing_emb = 0

    for repo_name in tqdm(idx):
        if repo_name not in emb_dict:
            missing_emb += 1
            continue
        ret[idx[repo_name]] = emb_dict[repo_name]

    print(f'Missing {missing_emb} embeddings.')
    # Try to convert `ret` into torch.FloatTensor. Failed.
    # It (OS) says, 'Killed'.
    # Confused. Maybe related to the current high CPU and MEM usage? 
    # ret = torch.FloatTensor(ret).to(device)
    # ret = torch.FloatTensor(ret.tolist()).to(device)  # 不懂为什么要 tolist
    return ret