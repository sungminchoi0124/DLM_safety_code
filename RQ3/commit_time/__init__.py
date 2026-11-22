"""RQ3: when does the refusal / compliance decision commit during dLLM decoding?

Modules
  capture   per-forward hidden-state trace, attached to any LLaDA decoder through hooks
  decode    standalone decoder: `quota` (= DiffuGuard vanilla path) and `threshold` (= Fast-dLLM rule)
  probe     outcome probe on the traced hidden states (D^2-Monitor style, retargeted to the response label)
  metrics   commit step s*, hesitation n_tau, textual commit, behavioural commit from resampling
  run_trace CLI: run_generate.py + trace
  resample  CLI: branch K continuations from intermediate states (behavioural commit)
  analyze   CLI: labels / fit / score / behavior

No module writes model text except the generation JSON files that run_generate.py / resample.py already
write for the judge (restricted output folder, never in git).
"""
