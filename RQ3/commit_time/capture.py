"""Per-forward trace of a LLaDA decoding run, recorded through hooks (decoder-agnostic).

The recorder sees every model forward while it is active. From the input ids of consecutive forwards it
derives which tracked positions were committed (or re-masked) at each step, and from layer hooks it stores
the mean hidden state over the tracked positions. It works for the original DiffuGuard generate(), the
Fast-dLLM no-cache generate(), and commit_time.decode.decode(). KV-cache decoders that feed only part of the
sequence are rejected (the pooled positions would change meaning).

Index convention
  forward f = 0..T-1 sees state x_f (before the commits of step f); x_T = final output.
  hidden[f]            pooled hidden state of forward f          [T, n_layers, d]
  hidden_infill[f]     same, pooled over the tracked prompt masks only (DIJA infill); zeros if none
  hidden_tail[f]       same, pooled over the tracked response positions only (assistant answer)
  n_masks[f]           tracked positions still masked in x_f     [T+1]
  n_commit[f]          tracked positions committed by step f      [T]   (masked in x_f, unmasked in x_{f+1})
  n_remask[f]          tracked positions re-masked by step f      [T]   (DiffuGuard repair)
  lead_unmasked[f]     of the first `lead_k` response positions, how many are unmasked in x_f   [T+1]
  cue_accept/refuse[f] committed tokens at step f inside the lead window that belong to the DiffuGuard
                       accept / refuse cue sets (counts only; no token ids are stored)              [T]

Logit statistics (logit_stats=True; from the model output of forward f, row 0, so before any cfg mixing):
  gap_all[f]           mean top-2 logit gap over the lead window (= Prophet answer region)          [T]
  gap_committed[f]     same, lead positions already unmasked in x_f (NaN if none)                   [T]
  gap_masked[f]        same, lead positions still masked in x_f (NaN if none)                       [T]
  mass_refuse[f]       RAEC-style refusal mass: sum of softmax probability over the refusal token
                       set, averaged over the lead positions still masked in x_f (NaN if none)      [T]
  mass_comply[f]       same for the compliance token set                                            [T]

Infill window (DIJA prompts carry masks inside the user turn; the judged DIJA response is that infill):
  infill_unmasked[f]   of the first `lead_k` prompt-mask positions (positions < prompt_len), how many are
                       unmasked in x_f                                                               [T+1]
  <key>_infill[f]      the five logit statistics above over that infill window (with logit_stats)    [T]
  For prompts without masks the infill window is empty: counts are 0 and the statistics NaN.
Only these floats are stored, never token ids or probabilities per token.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

FINAL = "final"
DEFAULT_LAYERS = (7, 15, 23, FINAL)
LOGIT_KEYS = ("gap_all", "gap_committed", "gap_masked", "mass_refuse", "mass_comply")
INFILL_SUFFIX = "_infill"
ALL_LOGIT_KEYS = LOGIT_KEYS + tuple(k + INFILL_SUFFIX for k in LOGIT_KEYS)


def parse_layers(spec: str | Sequence) -> list:
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for s in spec:
        out.append(FINAL if str(s) == FINAL else int(s))
    return out


def _transformer(model):
    core = getattr(model, "model", model)
    tr = getattr(core, "transformer", None)
    if tr is None:
        raise ValueError("expected a LLaDA model with model.model.transformer (trust_remote_code modeling_llada)")
    return tr


def _blocks(tr):
    if hasattr(tr, "blocks"):
        return list(tr.blocks)
    if hasattr(tr, "block_groups"):  # block_group_size > 1
        return [b for g in tr.block_groups for b in g]
    raise ValueError("no transformer.blocks / block_groups found")


def resolve_layer_modules(model, layers: Iterable) -> dict:
    """{name: module}; int i -> output of block i (= hidden_states[i+1]); 'final' -> ln_f (= hidden_states[-1])."""
    tr = _transformer(model)
    blocks = _blocks(tr)
    out = {}
    for l in layers:
        if l == FINAL:
            out[FINAL] = tr.ln_f
        else:
            if not 0 <= int(l) < len(blocks):
                raise ValueError(f"layer {l} out of range (model has {len(blocks)} blocks)")
            out[f"b{int(l)}"] = blocks[int(l)]
    return out


class TraceRecorder:
    """Context manager. `prompt_len` = length of the tokenized prompt (start of the response tail)."""

    def __init__(self, model, *, mask_id: int, prompt_len: int, layers=DEFAULT_LAYERS, lead_k: int = 16,
                 accept_ids: Optional[set] = None, refuse_ids: Optional[set] = None,
                 logit_stats: bool = False, mass_refuse_ids: Optional[set] = None,
                 mass_comply_ids: Optional[set] = None):
        self.model = model
        self.mask_id = int(mask_id)
        self.prompt_len = int(prompt_len)
        self.layer_mods = resolve_layer_modules(model, parse_layers(layers) if isinstance(layers, str) else layers)
        self.layer_names = list(self.layer_mods)
        self.lead_k = int(lead_k)
        self.accept = torch.tensor(sorted(accept_ids or []), dtype=torch.long)
        self.refuse = torch.tensor(sorted(refuse_ids or []), dtype=torch.long)
        self.logit_stats = bool(logit_stats)
        self.mass_ref = torch.tensor(sorted(mass_refuse_ids or []), dtype=torch.long)
        self.mass_cmp = torch.tensor(sorted(mass_comply_ids or []), dtype=torch.long)
        self._handles = []
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.tracked = None        # LongTensor of positions
        self.lead_pos = None       # LongTensor of the first lead_k response positions (subset of tracked)
        self.infill_pos = None     # LongTensor of the first lead_k prompt-mask positions (subset of tracked)
        self.seq_len = None
        self.prev_ids = None
        self.hidden = []           # list of [n_layers, d] float16 arrays (all tracked positions)
        self.hidden_infill = []    # same, prompt-mask positions only
        self.hidden_tail = []      # same, response positions only
        self._cur = {}
        self._cur_infill = {}
        self._cur_tail = {}
        self._infill_all = None    # LongTensor: tracked positions < prompt_len
        self._tail_all = None      # LongTensor: tracked positions >= prompt_len
        self.n_masks, self.n_commit, self.n_remask = [], [], []
        self.lead_unmasked, self.cue_accept, self.cue_refuse = [], [], []
        self.infill_unmasked = []
        self.logit = {k: [] for k in ALL_LOGIT_KEYS}
        self._lead_masked = None
        self._infill_masked = None
        self.finished = False

    def __enter__(self):
        self.reset()
        self._handles.append(self.model.register_forward_pre_hook(self._pre, with_kwargs=True))
        if self.logit_stats:
            self._handles.append(self.model.register_forward_hook(self._post))
        for name, mod in self.layer_mods.items():
            self._handles.append(mod.register_forward_hook(self._make_layer_hook(name)))
        return self

    def __exit__(self, *exc):
        self._flush()
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # ------------------------------------------------------------------ hooks
    def _pre(self, _module, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is None:
            raise RuntimeError("TraceRecorder: forward called without input_ids")
        ids = ids[:1].detach()  # cfg doubles the batch; the conditional row is row 0
        self._flush()
        if self.tracked is None:
            self.seq_len = ids.shape[1]
            self.tracked = torch.nonzero(ids[0] == self.mask_id, as_tuple=False).squeeze(1).cpu()
            resp = self.tracked[self.tracked >= self.prompt_len]
            self.lead_pos = resp[: self.lead_k]
            self.infill_pos = self.tracked[self.tracked < self.prompt_len][: self.lead_k]
            self._infill_all = self.tracked[self.tracked < self.prompt_len]
            self._tail_all = self.tracked[self.tracked >= self.prompt_len]
        elif ids.shape[1] != self.seq_len:
            raise RuntimeError("TraceRecorder: partial-sequence forward (KV cache?) is not supported; "
                               "run the trace with use_cache=False / dual_cache=False")
        cur = ids[0].to("cpu", copy=True)  # the decoder edits x in place
        if self.prev_ids is not None:
            self._transition(self.prev_ids, cur)
        self.n_masks.append(int((cur[self.tracked] == self.mask_id).sum()))
        self.lead_unmasked.append(int((cur[self.lead_pos] != self.mask_id).sum()))
        self._lead_masked = cur[self.lead_pos] == self.mask_id
        self.infill_unmasked.append(int((cur[self.infill_pos] != self.mask_id).sum()))
        self._infill_masked = cur[self.infill_pos] == self.mask_id
        self.prev_ids = cur
        self._cur, self._cur_infill, self._cur_tail = {}, {}, {}

    def _post(self, _module, _args, out):
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, (tuple, list)):
            logits = out[0]
        if logits is None or logits.dim() != 3 or logits.shape[1] != self.seq_len:
            raise RuntimeError("TraceRecorder: logit_stats needs full-sequence logits from the model forward")
        for pos, masked, suffix in ((self.lead_pos, self._lead_masked, ""),
                                    (self.infill_pos, self._infill_masked, INFILL_SUFFIX)):
            vals = self._window_stats(logits, pos, masked)
            for k in LOGIT_KEYS:
                self.logit[k + suffix].append(vals[k])

    def _window_stats(self, logits, pos, masked) -> dict:
        lg = logits[0, pos.to(logits.device)].float()                    # [k, V]
        masked = masked.to(logits.device)
        nan = float("nan")
        if lg.shape[0] == 0:
            vals = dict.fromkeys(LOGIT_KEYS, nan)
        else:
            top2 = torch.topk(lg, k=2, dim=-1).values
            gap = top2[:, 0] - top2[:, 1]
            vals = {"gap_all": float(gap.mean()),
                    "gap_committed": float(gap[~masked].mean()) if bool((~masked).any()) else nan,
                    "gap_masked": float(gap[masked].mean()) if bool(masked.any()) else nan}
            probs = torch.softmax(lg[masked], dim=-1) if bool(masked.any()) else None
            for key, ids in (("mass_refuse", self.mass_ref), ("mass_comply", self.mass_cmp)):
                if probs is None or len(ids) == 0:
                    vals[key] = nan
                else:
                    vals[key] = float(probs[:, ids.to(probs.device)].sum(dim=-1).mean())
        return vals

    def _make_layer_hook(self, name):
        def hook(_module, _inputs, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            if h.dim() != 3 or self.tracked is None or h.shape[1] != self.seq_len:
                return
            hf = h[0].float()

            def pool(idx):
                if idx is None or idx.numel() == 0:
                    return np.zeros(hf.shape[-1], np.float16)
                return hf[idx.to(h.device)].mean(dim=0).detach().cpu().numpy().astype(np.float16)
            self._cur[name] = pool(self.tracked)
            self._cur_infill[name] = pool(self._infill_all)
            self._cur_tail[name] = pool(self._tail_all)
        return hook

    def _flush(self):
        if self._cur:
            missing = [n for n in self.layer_names if n not in self._cur]
            if missing:
                raise RuntimeError(f"TraceRecorder: layers {missing} produced no output in this forward")
            self.hidden.append(np.stack([self._cur[n] for n in self.layer_names]))
            self.hidden_infill.append(np.stack([self._cur_infill[n] for n in self.layer_names]))
            self.hidden_tail.append(np.stack([self._cur_tail[n] for n in self.layer_names]))
            self._cur, self._cur_infill, self._cur_tail = {}, {}, {}

    def _transition(self, prev, cur):
        tp, tc = prev[self.tracked], cur[self.tracked]
        committed = (tp == self.mask_id) & (tc != self.mask_id)
        remasked = (tp != self.mask_id) & (tc == self.mask_id)
        self.n_commit.append(int(committed.sum()))
        self.n_remask.append(int(remasked.sum()))
        lp, lc = prev[self.lead_pos], cur[self.lead_pos]
        new = lc[(lp == self.mask_id) & (lc != self.mask_id)]
        self.cue_accept.append(int(torch.isin(new, self.accept).sum()) if len(self.accept) else 0)
        self.cue_refuse.append(int(torch.isin(new, self.refuse).sum()) if len(self.refuse) else 0)

    # ------------------------------------------------------------------ results
    def finish(self, final_ids: torch.Tensor):
        """Call once with the decoder's final ids [1, L] (or [L]) after the last forward."""
        self._flush()
        cur = (final_ids[0] if final_ids.dim() == 2 else final_ids).detach().to("cpu", copy=True)
        if cur.shape[0] != self.seq_len:
            raise RuntimeError("final ids length differs from the traced sequence")
        self._transition(self.prev_ids, cur)
        self.n_masks.append(int((cur[self.tracked] == self.mask_id).sum()))
        self.lead_unmasked.append(int((cur[self.lead_pos] != self.mask_id).sum()))
        self.infill_unmasked.append(int((cur[self.infill_pos] != self.mask_id).sum()))
        self.finished = True
        return self

    def arrays(self) -> dict:
        if not self.finished:
            raise RuntimeError("call finish(final_ids) first")
        T = len(self.hidden)
        assert len(self.n_masks) == T + 1 and len(self.n_commit) == T, "trace bookkeeping mismatch"
        extra = {}
        if self.logit_stats:
            assert all(len(v) == T for v in self.logit.values()), "logit stats bookkeeping mismatch"
            extra = {k: np.array(v, np.float32) for k, v in self.logit.items()}
        return extra | {
            "hidden": np.stack(self.hidden) if T else np.zeros((0, len(self.layer_names), 0), np.float16),
            "hidden_infill": (np.stack(self.hidden_infill) if T
                              else np.zeros((0, len(self.layer_names), 0), np.float16)),
            "hidden_tail": np.stack(self.hidden_tail) if T else np.zeros((0, len(self.layer_names), 0), np.float16),
            "layer_names": np.array(self.layer_names),
            "n_masks": np.array(self.n_masks, np.int32),
            "n_commit": np.array(self.n_commit, np.int32),
            "n_remask": np.array(self.n_remask, np.int32),
            "lead_unmasked": np.array(self.lead_unmasked, np.int32),
            "infill_unmasked": np.array(self.infill_unmasked, np.int32),
            "cue_accept": np.array(self.cue_accept, np.int32),
            "cue_refuse": np.array(self.cue_refuse, np.int32),
            "n_tracked": np.int32(len(self.tracked)),
            "n_response_tracked": np.int32(int((self.tracked >= self.prompt_len).sum())),
            "lead_k": np.int32(len(self.lead_pos)),
            "infill_k": np.int32(len(self.infill_pos)),
            "n_infill_tracked": np.int32(int((self.tracked < self.prompt_len).sum())),
            "prompt_len": np.int32(self.prompt_len),
            "seq_len": np.int32(self.seq_len),
        }

    def save(self, path: Path, extra: Optional[dict] = None) -> dict:
        """`extra`: additional numeric arrays (e.g. prophet_* from decode()); never text or token ids."""
        a = self.arrays()
        for k, v in (extra or {}).items():
            arr = np.asarray(v)
            if arr.dtype.kind not in "biuf":
                raise TypeError(f"extra array {k} must be numeric")
            a[k] = arr
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **a)
        return {k: (int(v) if np.ndim(v) == 0 else None) for k, v in a.items() if np.ndim(v) == 0} | {
            "n_forwards": int(a["hidden"].shape[0]), "file": path.name}


def load_trace(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_meta_line(path: Path, rec: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
