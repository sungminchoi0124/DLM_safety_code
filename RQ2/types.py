"""Initial common decoder contract. Without 0/A/B/C/D/E are implemented."""
from dataclasses import dataclass, field
from typing import Any

THRESHOLDS = {"0": None, "A": 0.9, "B": 0.7, "C": 0.5, "D": 0.9, "E": 0.9}
CACHE_MODES = {c: ("prefix" if c == "D" else "dual" if c == "E" else "none") for c in THRESHOLDS}

@dataclass(frozen=True)
class DecodeConfig:
    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 0.5
    seed: int = 0
    mask_id: int = 126336
    accel: str = "0"
    threshold: float | None = None
    cache: str = "none"
    defense: str = "without"
    prompt_mask_policy: str = "first_block"

    def validate(self):
        supported = tuple((c, t, CACHE_MODES[c], "without") for c, t in THRESHOLDS.items())
        if (self.accel, self.threshold, self.cache, self.defense) not in supported:
            raise ValueError("Only Without 0/A/B/C/D/E are implemented; no silent fallback")
        if self.prompt_mask_policy != "first_block":
            raise ValueError("Unsupported prompt mask policy")
        if self.gen_length <= 0 or self.block_length <= 0 or self.steps <= 0:
            raise ValueError("Lengths and steps must be positive")
        if self.gen_length % self.block_length or self.steps % (self.gen_length // self.block_length):
            raise ValueError("Generation length and steps must divide into complete blocks")
        if self.steps > self.gen_length or self.temperature < 0:
            raise ValueError("Require steps <= gen_length and temperature >= 0")

@dataclass
class DecodeResult:
    token_ids: Any
    n_masks: int
    prompt_masks_remaining: int
    generation_masks_remaining: int
    gen_nfe: int
    audit_nfe: int = 0
    repair_nfe: int = 0
    wall_ms: float = 0.0
    blocks: list[dict] = field(default_factory=list)

    @property
    def total_nfe(self):
        return self.gen_nfe + self.audit_nfe + self.repair_nfe
