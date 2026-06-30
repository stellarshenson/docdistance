# Fix - E06 jina-colbert-v2 CPU NaN (H26)

E06 hypothesis H26 (trained ColBERT scorer) crashes intermittently in the ColBERT subprocess with `AssertionError: NaN in ColBERT source encode`. The jina-colbert-v2 remote code is numerically unstable on the CPU forward - the encode returns NaN nondeterministically per process. No single setting removes it; the working fix is rate-reduction plus a bounded retry.

## Symptom

- **Where** - `notebooks/experiments/E06-kj-trained-scorers-cpu.ipynb`, the ColBERT subprocess (`/tmp/e06_colbert.py`, run via `SCORER_PY`)
- **Error** - `AssertionError: NaN in ColBERT source encode`, or a downstream `FileNotFoundError: /tmp/e06_colbert_grids.json` when the subprocess aborts before writing
- **Model** - `jinaai/jina-colbert-v2` via pylate `models.ColBERT(..., trust_remote_code=True)`; the remote `xlm-roberta-flash-implementation` runs a custom-attention XLM-R on CPU (`flash_attn` is not installed, so the slow fallback path runs)
- **Intermittent, not deterministic** - the failure is per-process, not per-call: within one process every encode agrees (all finite or all NaN); across fresh processes the same config flips. Measured base rate ≈ 40-50% finite, so a single run fails roughly half the time

## Root cause

- **Denormal blow-up** - subnormal floats in the custom CPU attention propagate to NaN; whether a process hits it depends on per-process memory state, which is why it looks random
- **Online re-resolution** - with the hub reachable, `trust_remote_code` re-resolves the remote modules per process; that path is measurably worse than loading the frozen cache
- **Silent-masking risk** - a `nan_to_num` downstream would zero the grid and ship a wrong H26 verdict instead of failing

## Fix

No single setting is sufficient - the four rate-reducers below still leave ≈ 10% NaN, so a bounded process retry closes the gap. Measured finite rates: base ≈ 40-50%, `set_flush_denormal` ≈ 73%, `+ offline cache` ≈ 91%, `+ 8 retries` → failure ≈ 10⁻⁸.

- **`torch.set_flush_denormal(True)`** - flush subnormals to zero; the single biggest lever
- **`HF_HUB_OFFLINE=1`** in the subprocess - load the frozen cache, skip online re-resolution (requires the model already cached)
- **`config_kwargs={"use_flash_attn": False}` + `model_kwargs={"dtype": torch.float32}` + `batch_size=1`** - disable the flash path, avoid bf16, and encode one sequence at a time so no padded positions enter the forward
- **Bounded retry at the process level** - re-run the isolated subprocess until it writes a finite grid (8 attempts), `RuntimeError` if all NaN; the in-subprocess `assert torch.isfinite(...)` makes each bad attempt exit nonzero, never a silent zero

```python
COLBERT_SCRIPT = r'''
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch
torch.set_num_threads(16)
torch.set_flush_denormal(True)
from pylate import models as pmodels
cb = pmodels.ColBERT(model_name_or_path=COLBERT_REPO, trust_remote_code=True,
                     config_kwargs={"use_flash_attn": False}, model_kwargs={"dtype": torch.float32})
Sd = cb.encode(S, is_query=False, convert_to_tensor=True, batch_size=1)
assert all(torch.isfinite(s).all() for s in Sd), "NaN in ColBERT source encode"
...
'''
# parent: retry the subprocess until a finite grid lands
for cb_try in range(1, 9):
    if CB_GRIDS.exists(): CB_GRIDS.unlink()
    r = subprocess.run([SCORER_PY, "/tmp/e06_colbert.py"], capture_output=True, text=True)
    if r.returncode == 0 and CB_GRIDS.exists():
        break
else:
    raise RuntimeError("ColBERT encode NaN after 8 attempts")
```

## Verification

- **Probe** - loading jina-colbert-v2 and encoding a representative statement set across fresh processes: base ≈ 4/10 finite; with `set_flush_denormal` 11/15 finite; with offline + flush 10/11 finite
- **Pass signal** - the subprocess prints `colbert <label> <shape> finite True` per document and `COLBERT_DONE`, then writes `/tmp/e06_colbert_grids.json`; the notebook prints `ColBERT encode finite on attempt N`
- **Isolation** - the encode runs in its own process with `CUDA_VISIBLE_DEVICES=""`, so a NaN or crash cannot poison the notebook kernel
- **Prerequisite** - the offline setting needs jina-colbert-v2 (and its remote-code repos) already in the HF cache; a first-time run must populate the cache online before the offline path works

## Honest status

- **It is a workaround, not a root-cause fix** - the instability lives in the third-party remote kernel; the retry tolerates it rather than eliminating it
- **The H26 verdict is unaffected** - H26 is Killed at gate on fidelity (recall@3 ≈ 0.47 < 0.60), independent of the NaN; the fix only ensures the cell executes so the verdict can be recorded
- **If the retry budget is ever exhausted** - it fails loud (`RuntimeError`), never a silent zero grid
