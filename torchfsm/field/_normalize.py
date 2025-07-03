from .._type import SpatialTensor
import torch
from typing import Union,Literal

def normalize(
    u: Union[SpatialTensor["B C H ..."],SpatialTensor["B C H ..."]],
    normalize_mode:Literal["normal_distribution","-1_1","0_1"]
    ):
    if normalize_mode not in ["normal_distribution","-1_1","0_1"]:
        raise ValueError(f"normalize_mode must be one of ['normal_distribution','-1_1','0_1'], but got {normalize_mode}")
    if normalize_mode == "normal_distribution":
        u = u - u.mean(dim=[i for i in range(1,u.ndim)],keepdim=True)
        u = u / u.std(dim=[i for i in range(1,u.ndim)],keepdim=True)
        return u
    else:
        shape=[u.shape[0]]+[1]*(len(u.shape)-1)
        max_v=torch.max(u.view(u.size(0), -1), dim=1).values.view(shape)
        min_v=torch.min(u.view(u.size(0), -1), dim=1).values.view(shape)
        u = (u - min_v) / (max_v - min_v)
        if normalize_mode == "-1_1":
            u = u * 2 - 1
        return u