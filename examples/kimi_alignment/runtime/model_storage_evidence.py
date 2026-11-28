"""Read local FSDP storage and copy policy after the real checkpoint load."""
import hashlib
from collections import Counter

import torch


@torch.no_grad()
def model_storage(module):
    from verl.workers.engine.fsdp.streaming_loader import _local_spec
    from torch.distributed.fsdp import FSDPModule
    pins=Counter()
    gates={}
    for sub in module.modules():
        if not isinstance(sub,FSDPModule):
            continue
        state=sub._get_fsdp_state()
        group=state._fsdp_param_group
        if group is not None:
            for par in group.fsdp_params:
                if par.offload_to_cpu:
                    pins["pinned" if par.pin_memory else "pageable"]+=1
    for name,p in module.named_parameters():
        if not name.endswith((".self_attn.A_log",".self_attn.dt_bias")):
            continue
        s=_local_spec(name,p)
        buf=s.local_tensor.detach().cpu().contiguous()
        gates[name]=dict(dtype=str(buf.dtype),offset=list(s.global_offset),shape=list(buf.shape),
                         sha256=hashlib.sha256(buf.view(torch.uint8).numpy().tobytes()).hexdigest())
    return dict(offloaded_parameters_by_copy_policy=dict(pins),kda_gate_shards=gates)
