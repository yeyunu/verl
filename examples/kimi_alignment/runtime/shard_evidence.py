"""Record local expert placement after the actual streaming BF16 load."""
import hashlib,json,os,socket
from pathlib import Path
import torch

def record(engine,module,path,zero_init):
    from verl.workers.engine.fsdp.streaming_loader import _local_spec
    import torch.distributed as dist
    rows=[];samples={}
    for name,p in module.named_parameters():
        if not name.endswith('.experts.gate_up_proj'):continue
        s=_local_spec(name,p)
        mesh=getattr(p,'device_mesh',None)
        r=dict(name=name,global_shape=list(s.global_shape),local_shape=list(s.local_shape),
               offset=list(s.global_offset),dtype=str(s.local_tensor.dtype))
        if mesh is not None:
            r.update(mesh_axes=mesh.mesh_dim_names,mesh_shape=list(mesh.shape),coordinate=mesh.get_coordinate(),placements=[str(v) for v in p.placements])
        if not rows:
            value=s.local_tensor[0,0,:16].detach().cpu().contiguous().view(torch.uint8)
            samples[name]=hashlib.sha256(value.numpy().tobytes()).hexdigest()
        rows.append(r)
    from model_storage_evidence import model_storage
    role='ref' if engine.engine_config.forward_only else 'actor'
    dest=Path(os.environ['KIMI_ALIGNMENT_AUDIT_DIR'])/'loaded-shards'
    dest.mkdir(parents=True,exist_ok=True)
    r=dict(node=socket.gethostname(),role=role,rank=dist.get_rank(),world=dist.get_world_size(),
           checkpoint=str(path),zero_init=bool(zero_init),samples=samples,experts=rows,storage=model_storage(module))
    import inspect
    sources = {inspect.getsourcefile(expert._ep_forward) for name, expert in module.named_modules()
               if name.endswith(".block_sparse_moe.experts") and hasattr(expert, "_ep_forward")}
    r["expert_implementation"] = [
        {"file": source, "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()} for source in sorted(sources)
    ]
    (dest/f'{role}-rank{dist.get_rank():03d}.json').write_text(json.dumps(r))

@torch.no_grad()
def policy_samples(model):
    """Observe a spread of BF16 coordinates without gathering any distributed weight."""
    import re
    entries=[]
    for name,p in model.named_parameters():
        layer=re.search(r'language_model\.model\.layers\.(\d+)\.',name)
        if layer and name.endswith(('o_proj.weight','.experts.gate_up_proj','input_layernorm.weight')):
            entries.append((int(layer.group(1)),name,p))
    levels=sorted({level for level,_,_ in entries})
    chosen={levels[0],levels[len(levels)//2],levels[-1]} if levels else set()
    observations={}
    for level,name,p in entries:
        if level not in chosen:continue
        t=p.to_local().detach() if hasattr(p,'to_local') else p.detach()
        if not t.numel():continue
        count=min(32768,t.numel())
        flat=torch.arange(count,device=t.device,dtype=torch.int64)*(t.numel()-1)//max(count-1,1)
        coords=[]
        for size in reversed(t.shape):
            coords.append(flat%size)
            flat=flat//size
        observations[name]=t[tuple(reversed(coords))].to(torch.bfloat16).cpu()
    return observations

def record_update(engine,before,grad_norm):
    after=policy_samples(engine.module)
    index=getattr(engine,'_k31_observed_updates',0)+1
    engine._k31_observed_updates=index
    rows={}
    for name,old in before.items():
        new=after[name]
        rows[name]=dict(sampled=old.numel(),changed=int((old!=new).sum()),nonfinite=int((~torch.isfinite(new)).sum()),max_abs=float((new.float()-old.float()).abs().max()),
                        before=hashlib.sha256(old.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        after=hashlib.sha256(new.view(torch.uint8).numpy().tobytes()).hexdigest())
    dest=Path(os.environ['KIMI_ALIGNMENT_AUDIT_DIR'])/'policy-updates'
    dest.mkdir(parents=True,exist_ok=True)
    rank=torch.distributed.get_rank()
    (dest/f'update{index}-rank{rank:03d}.json').write_text(json.dumps(dict(index=index,rank=rank,grad_norm=float(grad_norm),shards=rows)))
