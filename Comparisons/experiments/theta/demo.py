#!/usr/bin/env python3

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, 
    # model_kwargs={"torch_dtype": torch.bfloat16}, 
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
    # cache_dir='/media/dell/disk/vkx/huggingface/hub/models--meta-llama--Meta-Llama-3-8B',
)
res = pipeline('Hey, how is it going today?')
# ask = input('>> ')
# res = pipeline(ask)
print(res)
