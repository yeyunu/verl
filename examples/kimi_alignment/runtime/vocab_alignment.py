"""Publish only the vocabulary rows owned by the receiving vLLM TP worker."""
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

MARKER = ".__verl_vocab_rows__."


def is_vocab_weight(name):
    return name.split(".")[-2:] in (["embed_tokens", "weight"], ["lm_head", "weight"])


def export_vocab_weight(name, param, device, layout):
    from verl.workers.engine.fsdp.kimi_packed_weights import _exchange_expert_shards

    tp = int(layout["tensor_parallel_size"])
    if not is_vocab_weight(name) or not isinstance(param, DTensor) or param.ndim != 2:
        raise ValueError(f"Expected a sharded Kimi vocabulary matrix: {name}")
    if (param.dtype != torch.bfloat16 or param.size(0) % tp or dist.get_world_size() % tp
            or param.device_mesh.mesh.numel() != dist.get_world_size()
            or not any(isinstance(p, Shard) and p.dim == 0 for p in param.placements)):
        raise ValueError(f"Incompatible vocabulary layout: {name}, {param.shape}, {param.placements}, TP={tp}")
    # The same bounded resharder used by packed experts partitions axis zero.
    # Treat one vocabulary row as [hidden, 1], without changing its storage.
    rows = _exchange_expert_shards(param.unsqueeze(-1), device, tp).squeeze(-1)
    count = param.size(0) // tp
    start = (dist.get_rank() % tp) * count
    assert rows.shape == (count, param.size(1))
    return f"{name}{MARKER}{start}", rows.detach()


def decode_weight(name, value):
    if MARKER not in name:
        return name, value
    original, start = name.rsplit(MARKER, 1)
    if not is_vocab_weight(original):
        raise ValueError(f"Unrecognized vocabulary matrix: {name}")
    value._verl_kimi_vocab_start = int(start)
    return original, value


def copy_vocab_rows(module, param, value):
    start = module.shard_indices.org_vocab_start_index
    end = module.shard_indices.org_vocab_end_index
    if (start != value._verl_kimi_vocab_start or end-start != value.size(0)
            or param.ndim != 2 or value.ndim != 2 or value.size(1) != param.size(1)
            or value.size(0) > param.size(0) or value.dtype != param.dtype or param.dtype != torch.bfloat16
            or getattr(param, "output_dim", None) != 0 or getattr(param, "packed_dim", None) is not None):
        raise ValueError(f"Vocabulary export does not match receiving shard [{start},{end}), {value.shape}")
    param[:value.size(0)].data.copy_(value)
    param[value.size(0):].data.fill_(0)
