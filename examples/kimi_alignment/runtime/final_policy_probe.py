"""Fixed causal histories scored after a real synchronized update, without another optimizer step."""
import hashlib
import json
import os
from pathlib import Path

import torch
from tensordict import TensorDict, NonTensorData, NonTensorStack
from verl.utils import tensordict_utils as tu


def make_scoring_batch(rows):
    numeric = {name: [] for name in ("input_ids", "position_ids", "prompts", "responses", "loss_mask", "response_mask", "rollout_log_probs")}
    for row in rows:
        prompt = torch.tensor(row["prompt"], dtype=torch.long)
        token = torch.tensor([row["token"]], dtype=torch.long)
        ids = torch.cat((prompt,token))
        numeric["input_ids"].append(ids)
        numeric["position_ids"].append(torch.arange(len(ids)))
        numeric["prompts"].append(prompt)
        numeric["responses"].append(token)
        numeric["loss_mask"].append(torch.ones_like(token))
        numeric["response_mask"].append(torch.ones_like(token))
        numeric["rollout_log_probs"].append(torch.tensor([row["serving"]], dtype=torch.float32))
    fields = {}
    for name, values in numeric.items():
        lens = torch.tensor([v.numel() for v in values],dtype=torch.int64)
        offsets = torch.nn.functional.pad(lens.cumsum(0), (1,0))
        fields[name] = torch.nested.nested_tensor_from_jagged(torch.cat(values), offsets,
                                                              min_seqlen=int(lens.min()), max_seqlen=int(lens.max()))
    td = TensorDict(fields,batch_size=[len(rows)])
    for name, values in {
        "uid": [r["key"] for r in rows], "rollout_key": [r["key"] for r in rows],
        "session_id": [r["key"] for r in rows], "index": list(range(len(rows))),
        "global_steps": [r["version"] for r in rows], "multi_modal_inputs": [{} for _ in rows],
    }.items():
        td[name] = NonTensorStack(*(NonTensorData(v) for v in values))
    tu.assign_non_tensor(td,temperature=1.,compute_loss=False,calculate_entropy=False)
    return td


def compare(rows, outputs):
    values = outputs["log_probs"].unbind()
    if len(rows) != len(values):
        raise ValueError("Incomplete fixed probe outputs")
    result = []
    for r, scores in zip(rows, values, strict=True):
        if scores.ndim!=1 or scores.numel()!=len(r["prompt"])+1:
            raise ValueError("Expected one causal logprob per physical text position")
        old = float(scores[-2])
        delta = abs(old-r["serving"])
        if not torch.isfinite(torch.tensor([old,r["serving"]])).all():
            raise ValueError("Nonfinite fixed probe score")
        result.append(dict(key=r["key"],version=r["version"],replica=r["replica"],position=len(r["prompt"])-1,
                           prompt_sha256=hashlib.sha256(torch.tensor(r["prompt"]).numpy().tobytes()).hexdigest(),
                           token=r["token"],rollout_logprob=r["serving"],actor_old_logprob=old,abs_diff=delta))
    return result


def run_after_publication(trainer, result_dir):
    import ray
    root = Path(result_dir)
    version = int(trainer.global_steps)
    assert version == trainer.total_training_steps and version > 0
    assert not trainer.config.algorithm.rollout_correction.bypass_mode
    fixed = json.loads(Path(os.environ["KIMI_ALIGNMENT_LONG_TRAJECTORY"]).with_name("single.json").read_text())[0]
    handles = trainer.llm_server_manager.server_handles
    assert len(handles) == len(trainer.llm_server_manager.get_replicas())
    world = int(trainer.config.actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fully_shard_parallel_size)
    if world % len(handles):
        raise ValueError("Probe rows cannot be partitioned over all serving replicas")
    histories = [fixed["prompt"] + fixed["ids"][:i % len(fixed["ids"])] for i in range(world//len(handles))]
    dest = root/"audit/post-update-fixed"
    dest.mkdir(parents=True,exist_ok=True)
    requests = []
    for i, prefix in enumerate(histories):
        for replica, server in enumerate(handles):
            key = f"alignment-final-v{version}-history{i}-replica{replica}"
            options = dict(max_tokens=1,temperature=1.,top_p=1.,top_k=-1,seed=4100+i,
                           repetition_penalty=1.,logprobs=True)
            future = server.generate.remote(prompt_ids=prefix,sampling_params=options,request_id=key)
            requests.append((i,replica,key,prefix,future))
    outputs = ray.get([item[-1] for item in requests])
    rows = []
    replica_values = {}
    for (i,replica,key,prefix,_),output in zip(requests,outputs,strict=True):
        if (output.extra_fields["global_steps"] != version or len(output.token_ids)!=1 or len(output.log_probs)!=1
                or output.stop_reason != "completed"):
            raise ValueError(f"Invalid serving response for {key}")
        token, score = output.token_ids[0],float(output.log_probs[0])
        if replica_values.setdefault(i,(token,score)) != (token,score):
            raise ValueError(f"Replicas disagree on fixed history {i}")
        rows.append(dict(key=key,version=version,replica=replica,prompt=prefix,token=token,serving=score))
    (dest/"serving.json").write_text(json.dumps(rows))
    trainer.checkpoint_manager.sleep_replicas()
    old = trainer.actor_rollout_wg.compute_log_prob(make_scoring_batch(rows))
    result = compare(rows,old)
    out = dict(serving_version=version, rows=result, different=sum(r["abs_diff"]!=0 for r in result),
               mean_abs=sum(r["abs_diff"] for r in result)/len(result),max_abs=max(r["abs_diff"] for r in result))
    (dest/"independent-score.json").write_text(json.dumps(out,indent=2))
    if out["different"]:
        raise ValueError("Final online policy differs on fixed causal histories")
