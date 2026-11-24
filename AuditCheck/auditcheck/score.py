"""DiffuGuard's hidden-state audit from the first forward only (no generation).

The audit of the RQ2 combined decoder (RQ2/src/combined.py, block 0 of a DiffuGuard run):
    audit_hidden = 1 - cos( mean over block-0 positions of the last-layer hidden state at step 0,
                            baseline_hidden )
    baseline_hidden = mean last-layer hidden state over the clean request through the chat template, no masks
                      (original models/jailbreakbench_llada.py compute_baseline_hidden)
    block-0 positions = the first block_length generated positions + every mask inside the prompt (DIJA <mask:N>
                        and the 36-mask tail the ASR pipeline appends to plain requests)
    unsafe if audit_hidden >= lambda (0.1), OR the logit conflict score of the LAST block-0 step >= lambda.

Step 0 is the same for every decoding rule (the input is prompt + all-mask generation), so this score is exactly
what a DiffuGuard run would compute, for any column. The logit path needs the last block-0 step and is not
computed here (in the RQ1 DIJA runs it was ~1e-6, never the trigger).

Two averaging regions are returned:
    A : block 0 as above (RQ2 combined decoder, block 32; also RQ1 GSM8K via semi_ar, block 32)
    B : every generated position + the prompt masks (original generate() with one block: RQ1 DIJA, gen = block 128)
Plus per-layer means for a linear-probe comparison: over region A ("slots") and over the non-mask prompt tokens
("prompt").
"""
from __future__ import annotations

import sys

import torch

from .sets import REPO

for _p in (REPO, REPO / "RQ2", REPO / "DiffuGuard"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src import combined  # noqa: E402  (RQ2/src/combined.py)

MASK_ID = combined.MASK_ID
LAYOUTS = {"asr": dict(gen_length=128, block_length=32), "gsm5shot": dict(gen_length=256, block_length=32)}
cosine_distance = combined.cosine_distance


def user_text(jbl, item, mask_counts):
    """Decoded user turn of an item, as the RQ1/RQ2 runners build it."""
    if item["layout"] == "gsm5shot":
        return item["vanilla"]                      # run_gsm8k.py: the context as is, no process_user_text
    return jbl.process_user_text(item["refined"] or item["vanilla"], mask_counts)


def build_input(jbl, tokenizer, item, *, mask_counts, is_instruct, device):
    prompt = jbl.build_chat_prompt(tokenizer, user_text(jbl, item, mask_counts), is_instruct, None)
    return jbl.get_tokenized_input(prompt, tokenizer, device)


def baseline_len(jbl, tokenizer, item, is_instruct):
    prompt = jbl.build_chat_prompt(tokenizer, jbl.process_user_text(item["vanilla"], 0), is_instruct, None)
    return len(tokenizer(prompt)["input_ids"])


@torch.no_grad()
def first_forward(model, prompt, attention_mask=None, *, gen_length, block_length, mask_id=MASK_ID, layers=()):
    """Step-0 forward of a DiffuGuard decode. Returns the region means of the last layer (model dtype, as the
    decoder) and float32 CPU features of the requested hidden_states indices (0 = embeddings)."""
    device = next(model.parameters()).device
    P = prompt.shape[1]
    x = torch.cat([prompt.to(device), torch.full((1, gen_length), mask_id, dtype=torch.long, device=device)], dim=1)
    am = None
    if attention_mask is not None:  # combined.generate: pad with ones up to len(x)
        am = attention_mask.to(device)
        am = am if am.dtype == torch.bool else am != 0
        if am.shape[1] < x.shape[1]:
            am = torch.cat([am, torch.ones(1, x.shape[1] - am.shape[1], dtype=torch.bool, device=device)], dim=1)
        am = am[:, : x.shape[1]]
    out = model(x, attention_mask=am, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    if not hs:
        raise RuntimeError("the model returned no hidden_states")
    L = x.shape[1]
    pm = x[0, :P] == mask_id
    rA = torch.zeros(L, dtype=torch.bool, device=device)
    rA[P:P + block_length] = True
    rA[:P] |= pm
    rB = torch.zeros(L, dtype=torch.bool, device=device)
    rB[P:] = True
    rB[:P] |= pm
    rP = torch.zeros(L, dtype=torch.bool, device=device)
    rP[:P] = ~pm
    last = hs[-1]
    feats = {}
    for l in layers:
        if not -len(hs) <= l < len(hs):
            raise ValueError(f"layer {l} out of range (hidden_states has {len(hs)} entries)")
        feats[f"L{l}_slots"] = hs[l][:, rA, :].mean(dim=1).squeeze(0).float().cpu()
        feats[f"L{l}_prompt"] = hs[l][:, rP, :].mean(dim=1).squeeze(0).float().cpu()
    return {"mean_A": last[:, rA, :].mean(dim=1).squeeze(0).detach(),
            "mean_B": last[:, rB, :].mean(dim=1).squeeze(0).detach(),
            "n_prompt": int(P), "n_prompt_masks": int(pm.sum()), "n_A": int(rA.sum()), "n_B": int(rB.sum()),
            "n_hidden_states": len(hs), "feats": feats}


def audit_scores(ff, baseline_hidden):
    return {"audit_A": cosine_distance(ff["mean_A"], baseline_hidden),
            "audit_B": cosine_distance(ff["mean_B"], baseline_hidden)}
