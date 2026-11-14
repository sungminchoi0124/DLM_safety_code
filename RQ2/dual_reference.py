"""Independent staged E reference: original first block + original Dual kernel.

The first-block function is the pinned no-cache source with only the outer block-loop range
restricted to one iteration. Full sequence length and all future masks are preserved.
After it, the unmodified upstream Prefix generator handles the remaining three blocks.
This explicitly tests the C-defined first-block exception, rather than claiming E uses
Dual cache from block zero as the upstream all-Prefix generator does.
"""
import ast
import inspect
import torch


def dual_after_first_reference(model, prompt, config, upstream):
    if (prompt == config.mask_id).any():
        raise ValueError('Staged reference requires a mask-free prompt')
    source = inspect.getsource(upstream.generate)
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    loops = [n for n in ast.walk(function) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'num_block']
    if len(loops) != 1:
        raise RuntimeError('Upstream block-loop structure changed')
    loops[0].iter = ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Constant(value=1)], keywords=[])
    function.name = 'original_first_block'
    ast.fix_missing_locations(tree)
    namespace = dict(upstream.__dict__)
    exec(compile(tree, 'Fast-dLLM original generate [first-block range only]', 'exec'), namespace)
    x, first_nfe = namespace['original_first_block'](
        model, prompt, steps=config.steps, gen_length=config.gen_length,
        block_length=config.block_length, temperature=config.temperature,
        mask_id=config.mask_id, threshold=config.threshold)
    remaining = config.gen_length-config.block_length
    if not remaining:
        return x, first_nfe
    prefix = x[:, :prompt.shape[1]+config.block_length]
    out, nfe = upstream.generate_with_dual_cache(
        model, prefix, steps=config.steps-config.steps//(config.gen_length//config.block_length),
        gen_length=remaining, block_length=config.block_length, temperature=config.temperature,
        mask_id=config.mask_id, threshold=config.threshold)
    return out, first_nfe+nfe
