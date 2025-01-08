'''
Copyright (2024) Hefei University of Technology. 
Developers: liangtao shi, ting liu.

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
'''

import math
from typing import Callable, Tuple

import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np




def conditional_pooling(
    feat: torch.Tensor,
    threshold:float,
    window_size: Tuple[int, int],
) -> Tuple[Callable, Callable]:
    
    with torch.no_grad():
        
        
        ws_h, ws_w = int(window_size[0]), int(window_size[1])  #窗口尺寸,2*2
        stride_h, stride_w = ws_h, ws_w
        num_token_window = stride_h * stride_w #窗口内token数量,4
        
        x_cls, feat = feat[:, :1, :], feat[:, 1:, :] # 取出cls token之外的所有tokens,一共576个vision token
        B, N, D = feat.size()
        base_grid_H = int(math.sqrt(N))
        base_grid_W = base_grid_H
        assert base_grid_H * base_grid_W == N and base_grid_H % ws_h == 0 and base_grid_W % ws_w == 0

        feat = rearrange(feat, "b (h w) c -> b c h w", h=base_grid_H)
    
        feat = rearrange(feat, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H//ws_h, gw=base_grid_W//ws_w)
        b, gh, gw, c, ps_h, ps_w = feat.shape

        # Flatten mxm window for pairwise operations
        tensor_flattened = feat.reshape(b, gh, gw, c, -1)
    

        # Expand dims for pairwise operations
        tensor_1 = tensor_flattened.unsqueeze(-1)
        tensor_2 = tensor_flattened.unsqueeze(-2)

        # Compute cosine similarities
        sims = F.cosine_similarity(tensor_1, tensor_2, dim=3)

        # Exclude the self-similarity (i.e., similarity with oneself will be 1)
        sims_mask = 1 - torch.eye(ps_h * ps_w).to(sims.device)
        sims = sims * sims_mask

        # Average similarities (excluding the self-similarity)
        similarity_map = sims.sum(-1).sum(-1) / ((ps_h * ps_w) * (ps_h * ps_w - 1))
            
        similarity_map = rearrange(similarity_map.unsqueeze(1), 'b c h w-> b (c h w)')
        
        #--- adaptive section ---#
     
        n_B, n_H = similarity_map.shape
        node_mean = torch.tensor(threshold).cuda(sims.device)
        node_mean=node_mean.repeat(1,n_H)
        r = torch.ge(similarity_map, node_mean).sum(dim=1).min()
        # -------------# 
    
        #   get top k similar super patches 
        _, sim_super_patch_idxs = similarity_map.topk(r,dim=-1)
    
        # --- creating the mergabel and unmergable super  pathes
        tensor = torch.arange(base_grid_H * base_grid_W).reshape(base_grid_H, base_grid_W).to(feat.device)

        # Repeat the tensor to create a batch of size 2
        tensor = tensor.unsqueeze(0).repeat(B, 1, 1)
        

        # Apply unfold operation on last two dimensions to create the sliding window
        windowed_tensor = tensor.unfold(1, ws_h, stride_h).unfold(2, ws_w, stride_w)

        # Reshape the tensor to the desired shape 
        windowed_tensor = windowed_tensor.reshape(B, -1, num_token_window)
    
        # Use torch.gather to collect the desired elements
        gathered_tensor = torch.gather(windowed_tensor, 1, sim_super_patch_idxs.unsqueeze(-1).expand(-1, -1, num_token_window))


        # Create a mask for all indices, for each batch
        mask = torch.ones((B, windowed_tensor.shape[1]), dtype=bool).to(feat.device)

        # Create a tensor that matches the shape of indices and fill it with False
        mask_values = torch.zeros_like(sim_super_patch_idxs, dtype=torch.bool).to(feat.device)

        # Use scatter_ to update the mask. This will set mask[b, indices[b]] = False for all b
        mask.scatter_(1, sim_super_patch_idxs, mask_values)

        # Get the remaining tensor
        remaining_tensor = windowed_tensor[mask.unsqueeze(-1).expand(-1, -1, num_token_window)].reshape(B, -1, num_token_window)
        unm_idx = remaining_tensor.reshape(B, -1).sort(dim=-1).values.unsqueeze(-1)
        dim_index = (num_token_window)- 1 
        src_idx= gathered_tensor[:, :, :dim_index].reshape(B, -1).unsqueeze(-1)
        dst_idx= gathered_tensor[:, :, dim_index].reshape(B, -1).unsqueeze(-1)
        merge_idx = torch.arange(src_idx.shape[1]//dim_index).repeat_interleave(dim_index).repeat(B, 1).unsqueeze(-1).to(feat.device)


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
       # TODO: num_token_window can be undefined
       
        x_cls , x_feat =  x[:, :1, :], x[:, 1:, :]
        n, t1, c = x_feat.shape
        src = x_feat.gather(dim=-2, index=src_idx.expand(n, r*dim_index, c))
        dst = x_feat.gather(dim=-2, index=dst_idx.expand(n, r, c))
        unm = x_feat.gather(dim=-2, index=unm_idx.expand(n, t1 - (r*num_token_window), c))
        dst = dst.scatter_reduce(-2, merge_idx.expand(n,r*dim_index, c), src, reduce=mode)
        x = torch.cat([dst, unm], dim=1)
        x = torch.cat((x_cls, x), dim=1)
        return x

    return merge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")    
    x = x / size
    
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:

    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source