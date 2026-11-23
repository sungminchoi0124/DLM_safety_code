import ast
from dataclasses import replace
from pathlib import Path
import re
from types import SimpleNamespace
import pytest
import torch
from RQ2.data import REPO, load_items, prepare
from RQ2.decoding import generate_fast
from RQ2.types import DecodeConfig
from RQ2.upstream import load_upstream

class FakeModel(torch.nn.Module):
    device = torch.device("cpu")
    def forward(self, x):
        pos = torch.arange(x.shape[1])[None, :, None]
        voc = torch.arange(32)[None, None, :]
        logits = torch.cos((x[:, :, None] + x.sum() + pos + voc * 3).float())
        logits[:, :, 31] = -torch.inf
        return SimpleNamespace(logits=logits)

@pytest.fixture(scope="module")
def upstream():
    return load_upstream("/root/Fast-dLLM")

@pytest.mark.parametrize("temp", [0, 0.5])
@pytest.mark.parametrize("block", [8, 32])
def test_original_tokens_and_nfe_match(upstream, temp, block):
    model = FakeModel()
    prompt = torch.tensor([[3, 5, 7, 9]])
    cfg = DecodeConfig(steps=32, gen_length=32, block_length=block, temperature=temp, mask_id=31)
    torch.manual_seed(0)
    expected, nfe = upstream.generate(model, prompt, steps=32, gen_length=32, block_length=block,
                                      temperature=temp, mask_id=31)
    torch.manual_seed(0)
    result = generate_fast(model, prompt, cfg, upstream)
    assert torch.equal(result.token_ids, expected)
    assert result.total_nfe == nfe == 32

@pytest.mark.parametrize("prompt_masks", [1, 45, 100, 200])
def test_prompt_masks_finish_first_block_and_future_blocks_wait(upstream, prompt_masks):
    model = FakeModel()
    prompt = torch.tensor([[3] + [31] * prompt_masks + [5, 7]])
    seen = []
    handle = model.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    cfg = DecodeConfig(mask_id=31)
    try:
        result = generate_fast(model, prompt, cfg, upstream)
    finally:
        handle.remove()
    p = prompt.shape[1]
    assert len(seen) == 128
    assert (seen[32][:, :p] == 31).sum() == 0
    assert (seen[32][:, p:p+32] == 31).sum() == 0
    assert (seen[32][:, p+32:] == 31).sum() == 96
    assert torch.equal(result.token_ids[:, :p][prompt != 31], prompt[prompt != 31])
    assert [b["generation_masks_remaining"] for b in result.blocks] == [96, 64, 32, 0]
    assert result.n_masks == 0 and result.total_nfe == 128
    assert result.audit_nfe == result.repair_nfe == 0

@pytest.mark.parametrize("change", [dict(accel="A"), dict(accel="A", threshold=0.7), dict(threshold=0.9), dict(cache="prefix"),
                                         dict(defense="with"), dict(steps=127), dict(block_length=0)])
def test_no_silent_unsupported_conditions(change):
    with pytest.raises(ValueError):
        replace(DecodeConfig(), **change).validate()

def test_fixed_input_ids_and_pilot():
    items = load_items(REPO / "RQ2/data/ids/dija_100.txt")
    assert len(items) == 100
    assert [name for name, _, _ in items[:10]] == [f"dija-{i:03d}" for i in range(10)]
    expected = "".join(f"{name}\t{idx}\n" for name, idx, _ in items[:10])
    assert (REPO / "RQ2/data/ids/dija_pilot10.txt").read_text() == expected

def test_prompt_processing_matches_original_for_all_items():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO / "hf_models/LLaDA-8B-Instruct", local_files_only=True)
    source = REPO / "DiffuGuard/third_party/DiffuGuard/models/jailbreakbench_llada.py"
    tree = ast.parse(source.read_text())
    wanted = {"process_user_text", "build_chat_prompt", "pick_two_prompt_fields"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    scope = {"re": re, "MASK_TOKEN": "<|mdm_mask|>", "START_TOKEN": "<startoftext>",
             "END_TOKEN": "<endoftext>", "SPECIAL_TOKEN_PATTERN": r"<mask:(\d+)>"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    for _, _, item in load_items(REPO / "RQ2/data/ids/dija_100.txt"):
        actual, matching, vanilla, refined = prepare(item, tok)
        assert (vanilla, refined) == scope["pick_two_prompt_fields"](item)
        def legacy(text):
            return tok(scope["build_chat_prompt"](tok, scope["process_user_text"](text, 36), True),
                       return_tensors="pt")["input_ids"]
        expected, plain = legacy(refined), legacy(vanilla)
        assert torch.equal(actual, expected)
        assert matching == min(sum(a.item() == b.item() for a, b in zip(plain[0], expected[0])), len(plain[0]))
