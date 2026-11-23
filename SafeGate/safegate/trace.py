"""Step trace of block 0: what the decoder saw and committed at the response lead and the DIJA prompt masks.

Shared module (the per-step record proposed for the RAEC analysis): the decoder calls `record()` once per
forward of block 0, before it writes the step's commits, with tensors it already has. Nothing here draws random
numbers or changes the decoder's tensors, so the decoded tokens are the same with or without a trace
(tests/test_decoder.py checks it).

Tracked positions
  resp   : the first `resp_k` generated positions (P .. P+resp_k-1; all in block 0 when resp_k <= block_length)
  infill : every mask position inside the prompt (DIJA infill; empty for GSM8K)

One record per forward (phase "gen"), for each tracked group:
  masked  1 if the position is still a mask before this step's commits
  tok     the decoder's candidate token x0 (Gumbel-noised argmax at temperature > 0; the current token if the
          position is already committed)
  conf    the model confidence p(x0) the decoder compares with tau (float64 softmax in threshold mode)
  commit  1 if this step commits the position
  m       per-position probability mass over each token set: "<set>_ref" (refusal) and "<set>_cmp"
          (compliance), sum of p over the set's ids, for every tracked position (filter on `masked` to get the
          masses of open positions only, as RQ3 / RAEC average them)
plus k (positions the acceleration rule picked), n_commit (actually written; smaller than k when the gate held
some), and the gate's decision for the step (None without a gate).
Repair forwards (DiffuGuard rows) get a short record (phase "repair": commits only).

The records hold token ids, i.e. model text: write them only under a git-ignored outputs/ folder.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class StepTrace:
    def __init__(self, token_sets: dict, resp_k: int = 16, decimals: int = 4):
        self.token_sets = {name: (torch.tensor(r, dtype=torch.long), torch.tensor(c, dtype=torch.long))
                           for name, (r, c) in token_sets.items()}
        self.resp_k = int(resp_k)
        self.decimals = int(decimals)
        self.records = []
        self.resp_pos = self.infill_pos = None

    def begin(self, prompt: torch.Tensor, mask_id: int):
        """Called by the decoder before the first forward. prompt = [1, P] input ids."""
        P = prompt.shape[1]
        self.records = []
        self.prompt_len = P
        self.resp_pos = torch.arange(P, P + self.resp_k, device=prompt.device)
        self.infill_pos = torch.nonzero(prompt[0] == mask_id, as_tuple=False).squeeze(1)
        self._dev = {}

    def _ids(self, device):
        if device not in self._dev:
            self._dev[device] = {n: (r.to(device), c.to(device)) for n, (r, c) in self.token_sets.items()}
        return self._dev[device]

    def _r(self, t):
        return [round(float(v), self.decimals) for v in t.tolist()]

    def _group(self, pos, mask_index, x0, conf, probs_for, transfer_index):
        if pos.numel() == 0:
            return None
        g = {"masked": mask_index[0, pos].int().tolist(), "tok": x0[0, pos].tolist(),
             "conf": self._r(conf[0, pos]), "commit": transfer_index[0, pos].int().tolist(), "m": {}}
        if probs_for is not None:
            p = probs_for(pos)                                    # [n, V]
            for name, (r, c) in self._ids(p.device).items():
                g["m"][f"{name}_ref"] = self._r(p[:, r].sum(-1)) if r.numel() else None
                g["m"][f"{name}_cmp"] = self._r(p[:, c].sum(-1)) if c.numel() else None
        return g

    def record(self, *, phase, block, step, mask_index, x0, conf, probs_for, transfer_index, k, gate=None):
        rec = {"phase": phase, "block": int(block), "step": int(step), "k": int(k),
               "n_commit": int(transfer_index.sum())}
        if phase == "gen":
            rec["resp"] = self._group(self.resp_pos, mask_index, x0, conf, probs_for, transfer_index)
            rec["infill"] = self._group(self.infill_pos, mask_index, x0, conf, probs_for, transfer_index)
            rec["gate"] = gate
        else:
            rec["commit_resp"] = transfer_index[0, self.resp_pos].int().tolist()
            rec["commit_infill"] = (transfer_index[0, self.infill_pos].int().tolist()
                                    if self.infill_pos.numel() else [])
        self.records.append(rec)

    def write(self, path: Path, header: dict):
        """One jsonl file per item: a header line, then one line per step."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"header": header, "prompt_len": self.prompt_len, "resp_k": self.resp_k,
                                "n_infill": int(self.infill_pos.numel()), "n_steps": len(self.records)}) + "\n")
            for r in self.records:
                f.write(json.dumps(r) + "\n")


def read_trace(path: Path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]), [json.loads(l) for l in lines[1:] if l.strip()]
