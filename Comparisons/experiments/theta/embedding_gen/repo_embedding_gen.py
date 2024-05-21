#!/usr/bin/env python3

'''
原来的实验中, 仓库的向量组成是: 
E_{NL} ++ E_{CL} ++ E_{T} ++ E_{L}
(自然语言, 代码块, 主题, 语言)
后两者直接复用 ###### 中的结果, 只用 llama3 对前两者做嵌入. 

'''

import json
from tqdm import tqdm
from llama3_embedding import Llama3Embedder
from typing import * 
import pickle
import numpy as np 
import os
import torch
from utils import kv_emb_to_tensor

FILE_PREFIX = './data/repo_embedding'

class RepositoryEmbedding: 
    
    def __init__(self): 
        self.model = Llama3Embedder()

    def text_embedding(self) -> Dict[str, np.ndarray]:   # Array<4096>
        '''
        src_file => dst_file. 
        LIST <| JSON { text: string, project: string } => JSON <| { $project_name: Vector<4096> }
        '''
        src_file = f'{FILE_PREFIX}/text_src.txt'
        dst_file = f'{FILE_PREFIX}/text.pkl'

        if os.path.exists(dst_file): 
            print('"text.pkl" embedding file found, chosen.')
            with open(dst_file, 'rb') as fp: 
                return pickle.load(fp)
                
        def repo_seq() -> Iterable[Dict]: 
            with open(src_file) as fp: 
                for line in fp: 
                    yield json.loads(line)

        repo_text_lst = list(repo_seq())
        repo_text_embeddings = {
            it['project']: self.model(it['text'])
            for it in tqdm(repo_text_lst)
        }
        with open(dst_file, 'wb') as fp: 
            pickle.dump(repo_text_embeddings, fp)

        return repo_text_embeddings  # len(*): 49651

    # never tested
    def original_code_embedding(self) -> Dict[str, np.ndarray]:  # Array<768>
        '''
        考虑到新生成嵌入的时间实在是太长了, 
        所以先备用一个原来的嵌入的版本
        '''
        with open('data/repo_embedding/original_code.pkl', 'rb') as fp: 
            return pickle.load(fp)  # JSON <| { $repo_name: Array<768> }
    

    # warning: never tested        
    def new_code_embedding(self) -> Dict[str, np.ndarray]: 
        src_file = f'data/code_snippet/code_snippet.pkl'
        if not os.path.exists(src_file): 
            raise Exception(f'"{src_file}" not exists. generate new code_snippet embeddings first!')

        with open(src_file, 'rb') as fp: 
            data = pickle.load(fp)  # JSON <| { $project: { $path: Vector<4096> } }

        return {
            proj_k: np.average(list(proj_v.values()),axis=1)
            for proj_k, proj_v in data.items()
        }


    def discrete_embedding(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: 
        '''
        这块儿直接复用原来的结果就可以. 
        离散的数据有两部分:
        1. E_{T}: T is for Topic;
        2. E_{L}: L is for Language.
        它们也是以 JSON <| { $project_name: Vector<*> } 的形式组织的.
        '''
        topic_filepath    = f'{FILE_PREFIX}/topic.pkl'      # 256
        lanugage_filepath = f'{FILE_PREFIX}/language.pkl'   # 256

        with open(topic_filepath, 'rb') as fp: 
            topic_data = pickle.load(fp)

        with open(lanugage_filepath, 'rb') as fp:
            language_data = pickle.load(fp)

        return topic_data, language_data 

    def merged_embedding(
            self, 
            *,
            device=torch.device('cpu'),
            use_old_code_embedding: bool,
        ) -> torch.Tensor:
        '''
        注意, 这里的向量, 已经开始转换为编号了. 
        @see also: 
            * GNN/DataPreprocess/4.add_node_feature.py    
            * GNN/DataPreprocess/utils.py
        '''
        text_emb = self.text_embedding()
        topic_emb, lang_emb = self.discrete_embedding()
        code_emb = self.original_code_embedding() if use_old_code_embedding else self.new_code_embedding()

        print('numpy embeddings loaded.')

        def load_repo_idx() -> Dict[str, int]: 
            '''repo_name to repo_idx'''
            if not hasattr(self, '_repo_idx'): 
                with open(f'{FILE_PREFIX}/repo_idx.json') as fp: 
                    self._repo_idx = json.load(fp)
            return self._repo_idx
        repo_idx: Dict[str, int] = load_repo_idx()
        print(f'count of repo: {len(repo_idx)}')

        text_emb, topic_emb, lang_emb, code_emb = [
            torch.from_numpy(kv_emb_to_tensor(it, repo_idx, device=device))
            for it in [text_emb, topic_emb, lang_emb, code_emb]
        ] 
        # warning: missing embeddings. Total: 50,000; missing: 
        #   * text : 349
        #   * topic: 0
        #   * lang : 102
        #   * code : 4691 (???)

        return torch.cat([text_emb, topic_emb, lang_emb, code_emb], dim=1).to(device)


if __name__ == '__main__':
    klee = RepositoryEmbedding()
    filepath = f'{FILE_PREFIX}/embedding.pt'
    if os.path.exists(filepath): 
        print('embedding generated. Exit.')
        exit(0)
    repo_embedding = klee.merged_embedding(use_old_code_embedding=True) 
    torch.save(repo_embedding, filepath)
