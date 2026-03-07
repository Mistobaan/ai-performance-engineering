"""Explicit benchmark target helpers for FlashAttention-4 lab variants."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional


class FlashAttention4FixedConfigMixin:
    """Override env-selected lab config with explicit target settings."""

    fixed_mode: str
    fixed_backend: Optional[str] = None

    def __init__(self) -> None:
        super().__init__()
        updates = {"mode": self.fixed_mode}
        if self.fixed_backend is not None:
            updates["backend"] = self.fixed_backend
        self.config = replace(self.config, **updates)
