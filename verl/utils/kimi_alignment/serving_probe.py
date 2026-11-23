"""Submit opt-in same-policy Kimi serving checks from the agent loop."""
import json
import os
from pathlib import Path

async def replay_prefix(agent, request_id, prompt_ids, output_ids, multimodal):
    if os.environ.get("KIMI_ALIGNMENT_SERVING_PROBE") != "1" or len(output_ids) < 5:
        return
    directory = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "serving_invariance"
    directory.mkdir(exist_ok=True)
    try:
        with (directory / "source.json").open("x") as handle:
            json.dump({"request_id": request_id, "prompt_ids": list(prompt_ids),
                       "suffix":list(output_ids[:4]),"target":int(output_ids[4])}, handle)
    except FileExistsError:
        return
    await agent.server_manager.generate(
        request_id="alignment-probe-control", prompt_ids=list(prompt_ids),
        sampling_params={"temperature": 0., "max_tokens": 1}, **multimodal)
