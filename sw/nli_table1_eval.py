"""
NLI Table 1 Evaluation
========================
Reproduces Table 1 from the NLI paper:
  MMLU, GSM8k, HumanEval, Zero-shot Avg, Wikitext-2 Perplexity

Combines lm-eval harness for accuracy tasks and sliding-window PPL.

Usage:
    conda run -n vllm python nli_table1_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --mode both
    conda run -n vllm python nli_table1_eval.py --model Qwen/Qwen3-4B --mode both --device cuda:0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import gc
import argparse
import json
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from nli_engine import nli_forward, build_lut_from_paper, build_lut
from nli_eda_engine import eda_forward
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM

try:
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)
except Exception:
    pass  # uses cached credentials or public models
# ─────────────────────────────────────────────────────────────
#  NLI patching (reused)
# ─────────────────────────────────────────────────────────────

_NLI_CACHE = {}
_NLI_OPT_CACHE = {}

def _get_nli_lut(func_name, optimize=False, device='cuda', M_target=16, mantissa_bits=5, lut_bits=32):
    cache = _NLI_OPT_CACHE if optimize else _NLI_CACHE
    device_str = str(device)
    cache_key = f"{func_name}_{device_str}"
    # print(M_target)
    if cache_key not in cache:
        if optimize:
            from nli_wqs_pt import dp_cutpoint_search_aliens, get_function, get_domain
            func = get_function(func_name)
            domain = get_domain(func_name)
            cutpoints_list = dp_cutpoint_search_aliens(func, M_target=M_target, domain=domain, device='cuda', silent=True, mantissa_bits=mantissa_bits)
            cutpoints = torch.tensor(cutpoints_list, dtype=torch.float32)
            point_reg, mul_reg, lut_reg = build_lut(func, cutpoints, lut_bits)
        else:
            point_reg, mul_reg, lut_reg = build_lut_from_paper(func_name)
            
        point_reg = point_reg.to(device)
        mul_reg = mul_reg.to(device)
        lut_reg = lut_reg.to(device)
        cache[cache_key] = (point_reg, mul_reg, lut_reg)
    return cache[cache_key]

def nli_activation(x, func_name, optimize=False):
    p, m, l = _get_nli_lut(func_name, optimize=optimize, device=x.device)
    return nli_forward(x, p, m, l, fp16_hw=True)

def nli_softmax(x, dim=None, optimize=False, dtype=None):
    if dim is None:
        dim = -1
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max

    p, m, l = _get_nli_lut('exp', optimize=optimize, device=x.device)

    # Chunk along dim=0 to avoid OOM on large attention matrices
    numel = x_shifted.numel()
    CHUNK_THRESH = 2 * 1024 * 1024  # 2M elements
    if numel > CHUNK_THRESH and x_shifted.dim() >= 2:
        chunks = x_shifted.shape[0]
        exp_parts = []
        for c in x_shifted.chunk(max(1, chunks), dim=0):
            exp_parts.append(nli_forward(c, p, m, l, D_n=32, fp16_hw=True))
        exp_x = torch.cat(exp_parts, dim=0)
    else:
        exp_x = nli_forward(x_shifted, p, m, l, D_n=32, fp16_hw=True)

    exp_sum = exp_x.sum(dim=dim, keepdim=True)
    res = exp_x * nli_activation(exp_sum, 'reciprocal', optimize=optimize)
    if dtype is not None:
        res = res.to(dtype)
    return res

ACTIVATION_MAP = {
    nn.SiLU: 'silu',
    nn.GELU: 'gelu',
    nn.Sigmoid: 'sigmoid',
    nn.Tanh: 'tanh',
    nn.Hardswish: 'hardswish',
    nn.Mish: 'mish',
}

# transformers>=5.x uses its own activation wrappers (e.g. SiLUActivation)
try:
    from transformers.activations import (
        SiLUActivation, GELUActivation, MishActivation,
    )
    ACTIVATION_MAP[SiLUActivation] = 'silu'
    ACTIVATION_MAP[GELUActivation] = 'gelu'
    ACTIVATION_MAP[MishActivation] = 'mish'
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
#  EDA-NLI patching
# ─────────────────────────────────────────────────────────────

def eda_activation(x, func_name, max_lut=256, max_k=5, t_bits=None):
    return eda_forward(x, func_name, max_lut=max_lut, max_k=max_k, t_bits=t_bits)

def eda_softmax(x, dim=None, dtype=None, max_lut=256, max_k=5, t_bits=None):
    if dim is None:
        dim = -1
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max

    # Chunk along dim=0 to avoid OOM from eda_forward intermediate tensors
    numel = x_shifted.numel()
    CHUNK_THRESH = 2 * 1024 * 1024  # 2M elements
    if numel > CHUNK_THRESH and x_shifted.dim() >= 2:
        chunks = x_shifted.shape[0]
        exp_parts = []
        for c in x_shifted.chunk(max(1, chunks), dim=0):
            exp_parts.append(eda_forward(c, 'exp', max_lut=max_lut, max_k=max_k, t_bits=t_bits))
        exp_x = torch.cat(exp_parts, dim=0)
    else:
        exp_x = eda_forward(x_shifted, 'exp', max_lut=max_lut, max_k=max_k, t_bits=t_bits)

    exp_sum = exp_x.sum(dim=dim, keepdim=True)
    res = exp_x * eda_forward(exp_sum, 'reciprocal', max_lut=max_lut, max_k=max_k, t_bits=t_bits)
    if dtype is not None:
        res = res.to(dtype)
    return res

# ─────────────────────────────────────────────────────────────
#  HPLA patching
# ─────────────────────────────────────────────────────────────

def hpla_activation(x, func_name, max_lut=256):
    from nli_hpla_eval import hpla_forward
    return hpla_forward(x, func_name, max_lut=max_lut)

def hpla_softmax(x, dim=None, dtype=None, max_lut=256):
    from nli_hpla_eval import hpla_forward
    if dim is None:
        dim = -1
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max

    exp_x = hpla_forward(x_shifted, 'exp', max_lut=max_lut)
    exp_sum = exp_x.sum(dim=dim, keepdim=True)
    res = exp_x * hpla_forward(exp_sum, 'reciprocal', max_lut=max_lut)
    if dtype is not None:
        res = res.to(dtype)
    return res

def patch_model_hpla(model, max_lut=256):
    """Patch all supported Activations, RMSNorm, and Attention Softmax with HPLA approximations."""
    n_rms = n_act = n_softmax = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'RMSNorm' in module_type:
            def make_hpla_rmsnorm_forward(orig_module):
                def hpla_rmsnorm_forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    inv_rms = hpla_activation(variance + orig_module.variance_epsilon, 'rsqrt', max_lut=max_lut)
                    hidden_states = hidden_states * inv_rms
                    return (orig_module.weight * hidden_states).to(input_dtype)
                return hpla_rmsnorm_forward
            module.forward = make_hpla_rmsnorm_forward(module)
            n_rms += 1

        if hasattr(module, 'act_fn') and type(module.act_fn) in ACTIVATION_MAP:
            func_name = ACTIVATION_MAP[type(module.act_fn)]
            class HPLAActivationModule(nn.Module):
                def __init__(self, fn_name):
                    super().__init__()
                    self.fn_name = fn_name
                def forward(self, x):
                    return hpla_activation(x, self.fn_name, max_lut=max_lut)
            module.act_fn = HPLAActivationModule(func_name)
            n_act += 1

        if 'Attention' in module_type:
            orig_forward = module.forward
            def make_hpla_attention_forward(orig_module, orig_fwd):
                import sys
                try:
                    module_ns = sys.modules[orig_module.__module__]
                except KeyError:
                    module_ns = None

                def hpla_attention_forward(*args, **kwargs):
                    if module_ns and hasattr(module_ns, 'nn') and hasattr(module_ns.nn, 'functional'):
                        orig_softmax = module_ns.nn.functional.softmax
                        def custom_softmax(input, dim=None, _stacklevel=3, dtype=None):
                            return hpla_softmax(input, dim=dim, dtype=dtype, max_lut=max_lut)
                        module_ns.nn.functional.softmax = custom_softmax
                        try:
                            return orig_fwd(*args, **kwargs)
                        finally:
                            module_ns.nn.functional.softmax = orig_softmax
                    else:
                        import torch.nn.functional as F
                        orig_softmax = F.softmax
                        def custom_softmax(input, dim=None, _stacklevel=3, dtype=None):
                            return hpla_softmax(input, dim=dim, dtype=dtype, max_lut=max_lut)
                        F.softmax = custom_softmax
                        try:
                            return orig_fwd(*args, **kwargs)
                        finally:
                            F.softmax = orig_softmax

                return hpla_attention_forward
            module.forward = make_hpla_attention_forward(module, orig_forward)
            n_softmax += 1

    return n_rms, n_act, n_softmax


def patch_model_eda(model, max_lut=256, max_k=5, t_bits=None):
    """Patch all supported Activations, RMSNorm, and Attention Softmax with EDA-NLI approximations."""
    n_rms = n_act = n_softmax = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'RMSNorm' in module_type:
            def make_eda_rmsnorm_forward(orig_module):
                def eda_rmsnorm_forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    inv_rms = eda_activation(variance + orig_module.variance_epsilon, 'rsqrt', max_lut=max_lut, max_k=max_k, t_bits=t_bits)
                    hidden_states = hidden_states * inv_rms
                    return (orig_module.weight * hidden_states).to(input_dtype)
                return eda_rmsnorm_forward
            module.forward = make_eda_rmsnorm_forward(module)
            n_rms += 1

        if hasattr(module, 'act_fn') and type(module.act_fn) in ACTIVATION_MAP:
            func_name = ACTIVATION_MAP[type(module.act_fn)]
            class EDAActivationModule(nn.Module):
                def __init__(self, fn_name):
                    super().__init__()
                    self.fn_name = fn_name
                def forward(self, x):
                    return eda_activation(x, self.fn_name, max_lut=max_lut, max_k=max_k, t_bits=t_bits)
            module.act_fn = EDAActivationModule(func_name)
            n_act += 1

        if 'Attention' in module_type:
            orig_forward = module.forward
            def make_eda_attention_forward(orig_module, orig_fwd):
                import sys
                try:
                    module_ns = sys.modules[orig_module.__module__]
                except KeyError:
                    module_ns = None

                def eda_attention_forward(*args, **kwargs):
                    if module_ns and hasattr(module_ns, 'nn') and hasattr(module_ns.nn, 'functional'):
                        orig_softmax = module_ns.nn.functional.softmax
                        def custom_softmax(input, dim=None, _stacklevel=3, dtype=None):
                            return eda_softmax(input, dim=dim, dtype=dtype, max_lut=max_lut, max_k=max_k, t_bits=t_bits)
                        module_ns.nn.functional.softmax = custom_softmax
                        try:
                            return orig_fwd(*args, **kwargs)
                        finally:
                            module_ns.nn.functional.softmax = orig_softmax
                    else:
                        import torch.nn.functional as F
                        orig_softmax = F.softmax
                        def custom_softmax(input, dim=None, _stacklevel=3, dtype=None):
                            return eda_softmax(input, dim=dim, dtype=dtype, max_lut=max_lut, max_k=max_k, t_bits=t_bits)
                        F.softmax = custom_softmax
                        try:
                            return orig_fwd(*args, **kwargs)
                        finally:
                            F.softmax = orig_softmax

                return eda_attention_forward
            module.forward = make_eda_attention_forward(module, orig_forward)
            n_softmax += 1

    for label, count in [('RMSNorm', n_rms), ('Activations', n_act), ('Softmax', n_softmax)]:
        if count == 0:
            raise RuntimeError(
                f"patch failed: 0 {label} patched (rms={n_rms}, act={n_act}, softmax={n_softmax}). "
                f"Check model structure or transformers version ({__import__('transformers').__version__})."
            )
    return n_rms, n_act, n_softmax

def patch_model_nli(model, optimize=False):
    """Patch all supported Activations, RMSNorm, and Attention Softmax with NLI approximations."""
    n_rms = n_act = n_softmax = 0
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'RMSNorm' in module_type:
            def make_nli_rmsnorm_forward(orig_module):
                def nli_rmsnorm_forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    inv_rms = nli_activation(variance + orig_module.variance_epsilon, 'rsqrt', optimize=optimize)
                    hidden_states = hidden_states * inv_rms
                    return (orig_module.weight * hidden_states).to(input_dtype)
                return nli_rmsnorm_forward
            module.forward = make_nli_rmsnorm_forward(module)
            n_rms += 1
            
        if hasattr(module, 'act_fn') and type(module.act_fn) in ACTIVATION_MAP:
            func_name = ACTIVATION_MAP[type(module.act_fn)]
            class NLIActivationModule(nn.Module):
                def __init__(self, fn_name):
                    super().__init__()
                    self.fn_name = fn_name
                def forward(self, x):
                    return nli_activation(x, self.fn_name, optimize=optimize)
            module.act_fn = NLIActivationModule(func_name)
            n_act += 1
            
        if 'Attention' in module_type:
            orig_forward = module.forward
            def make_nli_attention_forward(orig_module, orig_fwd):
                # Dynamically locate the module namespace (e.g., transformers.models.llama.modeling_llama)
                import sys
                try:
                    module_ns = sys.modules[orig_module.__module__]
                except KeyError:
                    # Fallback if module name somehow isn't in sys.modules
                    module_ns = None

                def nli_attention_forward(*args, **kwargs):
                    if module_ns and hasattr(module_ns, 'nn') and hasattr(module_ns.nn, 'functional'):
                        orig_softmax = module_ns.nn.functional.softmax
                        def custom_softmax(input, dim=None, _stacklevel=3, dtype=None):
                            return nli_softmax(input, dim=dim, optimize=optimize, dtype=dtype)
                        module_ns.nn.functional.softmax = custom_softmax
                        try:
                            return orig_fwd(*args, **kwargs)
                        finally:
                            module_ns.nn.functional.softmax = orig_softmax
                    else:
                        # Fallback to direct PyTorch patch if namespace fails
                        import torch.nn.functional as F
                        orig_softmax = F.softmax
                        def custom_softmax(input, dim=None, _stacklevel=3, dtype=None):
                            return nli_softmax(input, dim=dim, optimize=optimize, dtype=dtype)
                        F.softmax = custom_softmax
                        try:
                            return orig_fwd(*args, **kwargs)
                        finally:
                            F.softmax = orig_softmax
                            
                return nli_attention_forward
            module.forward = make_nli_attention_forward(module, orig_forward)
            n_softmax += 1

    if n_rms > 0 and n_act == 0:
        raise RuntimeError(
            f"patch_model_nli: {n_rms} RMSNorm patched but 0 Activations — "
            f"likely missing activation class in ACTIVATION_MAP. "
            f"Check transformers version ({__import__('transformers').__version__})."
        )
    return n_rms, n_act, n_softmax


# ─────────────────────────────────────────────────────────────
#  Wikitext-2 Perplexity (sliding window)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_perplexity(model, tokenizer, max_length=None, stride=None):
    """Evaluate Wikitext-2 perplexity using sliding window.

    Automatically adapts max_length to the model's max_position_embeddings.
    Filters out nan/inf losses from individual windows.
    """
    from datasets import load_dataset
    from tqdm import tqdm
    import math

    # Auto-detect model's max context length
    if max_length is None:
        if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
            max_length = min(model.config.max_position_embeddings, 2048)
        else:
            max_length = 2048
    if stride is None:
        stride = max_length // 4

    print(f"      [PPL] max_length={max_length}, stride={stride}")

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])

    # Truncate tokenization to avoid "sequence length > max" warnings
    encodings = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=max_length * 256)  # enough for wikitext-2
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)
    device = next(model.parameters()).device
    first_device = model.model.embed_tokens.weight.device if hasattr(model, 'model') else device

    nlls = []
    prev_end_loc = 0

    start_locs = list(range(0, seq_len, stride))
    pbar = tqdm(start_locs, desc="    Perplexity Windows")

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break
            continue
        input_chunk = input_ids[:, begin_loc:end_loc].to(first_device)
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
        loss_val = outputs.loss.item()

        # Skip nan/inf losses (can happen at chunk boundaries)
        if not math.isfinite(loss_val):
            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break
            continue

        nlls.append(loss_val)

        if len(nlls) % 10 == 0 and nlls:
            current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
            pbar.set_postfix({'ppl': f"{current_ppl:.4f}"})

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    if not nlls:
        print("      [PPL] WARNING: no valid windows, returning nan")
        return float('nan')

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


# ─────────────────────────────────────────────────────────────
#  Table 1 task definitions
# ─────────────────────────────────────────────────────────────

# lm-eval tasks for accuracy benchmarks
ACCURACY_TASKS = ["mmlu", "gsm8k", "humaneval"]

# Zero-shot tasks (for computing average)
ZEROSHOT_TASKS = [
    "arc_challenge", "arc_easy", "boolq", "piqa",
    "hellaswag", "openbookqa", "lambada_openai", "winogrande",
]

TASK_METRIC = {
    "mmlu": "acc",
    "gsm8k": "exact_match",
    "humaneval": "pass@1",
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "boolq": "acc",
    "piqa": "acc_norm",
    "hellaswag": "acc_norm",
    "openbookqa": "acc_norm",
    "lambada_openai": "acc",
    "winogrande": "acc",
}


def extract_score(results_dict, task_name):
    """Extract the primary metric score from lm-eval results."""
    if task_name not in results_dict.get('results', {}):
        # For grouped tasks like mmlu, check if there's an aggregate
        for key in results_dict.get('results', {}):
            if task_name in key:
                metric = TASK_METRIC.get(task_name, 'acc')
                val = results_dict['results'][key].get(metric, None)
                if val is not None:
                    return val * 100 if val <= 1.0 else val
        return None

    task_results = results_dict['results'][task_name]
    metric = TASK_METRIC.get(task_name, 'acc')

    # Try primary metric
    val = task_results.get(metric, None)
    if val is None:
        # Try with comma suffix (lm-eval format)
        val = task_results.get(f"{metric},none", None)
    if val is None:
        # Fallback: any acc-like metric
        for k, v in task_results.items():
            if 'acc' in k and isinstance(v, (int, float)):
                val = v
                break
    if val is None:
        # Try pass@1 variants
        for k, v in task_results.items():
            if 'pass' in k and isinstance(v, (int, float)):
                val = v
                break

    if val is not None:
        return val * 100 if val <= 1.0 else val
    return None


# ─────────────────────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────────────────────

def _config_path(save_dir, model_name, config):
    return os.path.join(save_dir, f"table1_{model_name.replace('/', '_')}_{config}.json")


def _save_incremental(path, model_name, config, result):
    with open(path, 'w') as f:
        json.dump({'model': model_name, 'mode': config,
                   'table1': {config: result}}, f, indent=2, default=str)


def _load_partial(path, config):
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        return saved.get('table1', {}).get(config, {})
    return {}


def run_table1_eval(
    model_name: str,
    mode: str = 'both',
    device: str = 'cuda:0',
    dtype: str = 'auto',
    batch_size: int = 16,
    save_dir: str = '/home/jmw/ing/research/nli/nli_results',
    t_bits: int = None,
):
    """Run full Table 1 evaluation with incremental save and resume."""
    os.makedirs(save_dir, exist_ok=True)
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'auto': 'auto'}
    torch_dtype = dtype_map.get(dtype, 'auto')

    configs = []
    if mode in ('baseline', 'both', 'all'):
        configs.append('baseline')
    if mode in ('nli', 'both', 'all'):
        configs.append('nli')
    if mode in ('nli_opt', 'all'):
        configs.append('nli_opt')
    if mode in ('nli_eda', 'all'):
        configs.append('nli_eda')
    if mode in ('nli_hpa', 'all'):
        configs.append('nli_hpa')

    all_results = {}
    all_tasks = ACCURACY_TASKS + ZEROSHOT_TASKS

    for config in configs:
        cfg_path = _config_path(save_dir, model_name, config)
        result = _load_partial(cfg_path, config)
        done = set(result.get('_done', []))

        # Skip if fully complete
        if result.get('_complete'):
            print(f"\n  [RESUME] {config} already complete — skipping")
            all_results[config] = result
            continue

        print(f"\n{'='*70}")
        print(f"  TABLE 1 EVALUATION — {config.upper()}")
        print(f"  Model: {model_name}")
        if done:
            print(f"  Resuming — already done: {sorted(done)}")
        print(f"{'='*70}")

        # Load model
        print(f"\n  [1] Loading model ({torch_dtype})...")
        t0 = time.time()
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True,
            device_map="auto" if device == "auto" else device,
            attn_implementation="eager"
        )
        hf_model.eval()
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        # Apply patches
        if config == 'nli':
            print(f"  [2] Applying NLI patches (Paper Cutpoints)...")
            n_rms, n_act, n_soft = patch_model_nli(hf_model, optimize=False)
            print(f"      Patched: {n_rms} RMSNorm, {n_act} Activations, {n_soft} Softmax intercepts")
        elif config == 'nli_opt':
            print(f"  [2] Applying NLI patches (Optimized Zero-Multiplier Cutpoints)...")
            n_rms, n_act, n_soft = patch_model_nli(hf_model, optimize=True)
            print(f"      Patched: {n_rms} RMSNorm, {n_act} Activations, {n_soft} Softmax intercepts")
        elif config == 'nli_eda':
            print(f"  [2] Applying EDA-NLI patches (Exponent-Direct Addressing)...")
            n_rms, n_act, n_soft = patch_model_eda(hf_model, t_bits=t_bits)
            print(f"      Patched: {n_rms} RMSNorm, {n_act} Activations, {n_soft} Softmax intercepts (t_bits={t_bits})")
        elif config == 'nli_hpa':
            print(f"  [2] Applying HPLA patches (Hierarchical Piecewise Linear Approximation)...")
            n_rms, n_act, n_soft = patch_model_hpla(hf_model)
            print(f"      Patched: {n_rms} RMSNorm, {n_act} Activations, {n_soft} Softmax intercepts")

        # ── Wikitext-2 Perplexity ──
        if 'ppl' in done:
            ppl = result['wikitext2_ppl']
            print(f"\n  [3] PPL already done = {ppl:.2f}")
        else:
            print(f"\n  [3] Evaluating Wikitext-2 Perplexity...")
            ppl = eval_perplexity(hf_model, hf_tokenizer)
            print(f"      PPL = {ppl:.2f}")
            result['wikitext2_ppl'] = ppl
            done.add('ppl')
            result['_done'] = sorted(done)
            _save_incremental(cfg_path, model_name, config, result)

        # ── lm-eval accuracy tasks (one by one, save after each) ──
        print(f"\n  [4] Evaluating accuracy tasks via lm-eval...")
        lm = HFLM(
            pretrained=hf_model,
            tokenizer=hf_tokenizer,
            batch_size=batch_size,
        )

        from tqdm import tqdm
        pending = [t for t in all_tasks if t not in done]
        if pending:
            pbar = tqdm(pending, desc="    Accuracy Tasks")
            for task in pbar:
                res = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=[task],
                    batch_size=batch_size,
                    log_samples=False,
                    confirm_run_unsafe_code=True,
                )
                score = extract_score(res, task) if 'results' in res and task in res['results'] else None
                result.setdefault('zeroshot_detail', {})[task] = score
                result[task] = score
                # Show metric in tqdm
                score_str = f"{score:.2f}" if score is not None else "N/A"
                pbar.set_postfix_str(f"{task}={score_str}")
                done.add(task)
                result['_done'] = sorted(done)
                _save_incremental(cfg_path, model_name, config, result)
        else:
            print("    All tasks already done.")

        # ── Compute aggregates and finalize ──
        scores = {t: result.get(t) for t in all_tasks}
        result['mmlu'] = scores.get('mmlu')
        result['gsm8k'] = scores.get('gsm8k')
        result['humaneval'] = scores.get('humaneval')
        zs_scores = [scores[t] for t in ZEROSHOT_TASKS if scores.get(t) is not None]
        result['zeroshot_avg'] = sum(zs_scores) / len(zs_scores) if zs_scores else None
        result['zeroshot_detail'] = {t: scores.get(t) for t in ZEROSHOT_TASKS}
        result['_complete'] = True
        _save_incremental(cfg_path, model_name, config, result)

        all_results[config] = result
        elapsed = time.time() - t0
        print(f"\n  Done ({elapsed:.0f}s) — saved to {cfg_path}")

        del lm, hf_model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Print Table 1 format ──
    print(f"\n{'='*80}")
    print(f"  TABLE 1 RESULTS: {model_name}")
    print(f"{'='*80}")

    header = f"  {'Metric':20s}"
    for c in configs:
        header += f"  {c:>12s}"
    if len(configs) == 2:
        header += f"  {'Δ':>10s}"
    print(header)
    print(f"  {'-'*20}" + f"  {'-'*12}" * len(configs) +
          ("  " + "-"*10 if len(configs) == 2 else ""))

    metrics = [
        ('MMLU', 'mmlu', '↑'),
        ('GSM8k', 'gsm8k', '↑'),
        ('HumanEval', 'humaneval', '↑'),
        ('Zero-shot Avg', 'zeroshot_avg', '↑'),
        ('Wikitext-2 PPL', 'wikitext2_ppl', '↓'),
    ]

    for display, key, direction in metrics:
        row = f"  {display + ' (' + direction + ')':20s}"
        vals = []
        for c in configs:
            v = all_results[c].get(key)
            if v is not None:
                row += f"  {v:12.2f}"
                vals.append(v)
            else:
                row += f"  {'N/A':>12s}"
        if len(vals) == 2:
            diff = vals[1] - vals[0]
            row += f"  {diff:+10.2f}"
        print(row)

    print(f"{'='*80}\n")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLI Table 1 Evaluation')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--mode', type=str, default='nli_eda',
                        choices=['baseline', 'nli', 'nli_eda', 'both', 'all'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['float16', 'bfloat16', 'auto'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--save-dir', type=str, default='/home/jmw/ing/research/nli/nli_results')
    parser.add_argument('--t-bits', type=str, default='adaptive',
                        help='Interpolation bit-width: integer, "adaptive", or None=continuous')
    args = parser.parse_args()

    # Parse t_bits: int, 'adaptive', or None
    t_bits = args.t_bits
    if t_bits is not None and t_bits != 'adaptive':
        t_bits = int(t_bits)

    run_table1_eval(
        model_name=args.model,
        mode=args.mode,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        t_bits=t_bits,
    )
