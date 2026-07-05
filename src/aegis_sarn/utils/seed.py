'''Reproducibility helpers.'''

from __future__ import annotations

import random

import torch

from aegis_sarn.config import SeedConfig


def set_global_seed(config: SeedConfig | int) -> None:
    '''Seed Python and PyTorch for a reproducible Phase 1 run.'''

    seed_config = config if isinstance(config, SeedConfig) else SeedConfig(seed=config)
    random.seed(seed_config.seed)
    torch.manual_seed(seed_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_config.seed)
    torch.use_deterministic_algorithms(seed_config.deterministic_algorithms)

