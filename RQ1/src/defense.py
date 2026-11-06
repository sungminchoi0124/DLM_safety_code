"""DiffuGuard with progress schedule: alpha = alpha0 * (1 - committed / total masks); repair steps = max(1, round(0.125 * steps)). schedule: step = original formula (regression test T2-1). defense: none = low_confidence remasking, sp_mode off. Original code under DiffuGuard/third_party is NOT modified.

Owner: T2-2, T2-3 (2번). Contract: docs/interfaces.md.
"""


def __getattr__(name):
    raise NotImplementedError("T2-2, T2-3 (2번): src/defense.py is a stub")
