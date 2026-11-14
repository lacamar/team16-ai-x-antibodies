"""Loading IgLM's GPT-2 checkpoint for training rather than inference.

Upstream ``iglm.IgLM`` calls ``.eval()`` in its constructor and never exposes the
module for optimisation, so we load the checkpoint directly. The architecture is a
plain ``GPT2LMHeadModel`` (4 layers, d_model 512, 8 heads, vocab 33, ~12.9M params),
which full-finetunes comfortably on one H100.

Two checkpoint quirks are handled here:

* ``config.json`` carries ``bos_token_id``/``eos_token_id`` of 50256, inherited from
  GPT-2's vocabulary but out of range for IgLM's 33-token vocab. Left alone,
  transformers warns on every load and ``generate`` can misbehave. We repoint them at
  IgLM's real sentinels: [CLS] terminates an infill, [PAD] pads.
* The released weights contain ``transformer.h.*.attn.masked_bias`` buffers that
  modern transformers no longer defines. They are reported as UNEXPECTED and are
  genuinely unused -- the causal mask is rebuilt at runtime -- so ignoring them is safe.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import transformers

import iglm
from her2_iglm.tokenizer import IgLMTokenizer

TRAINED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(iglm.__file__)), "trained_models"
)

CHECKPOINTS = {
    "IgLM": os.path.join(TRAINED_MODELS_DIR, "IgLM"),
    "IgLM-S": os.path.join(TRAINED_MODELS_DIR, "IgLM-S"),
}


def resolve_checkpoint(name_or_path: str) -> str:
    """Accept either a bundled model name or a path to a saved checkpoint."""
    if name_or_path in CHECKPOINTS:
        return CHECKPOINTS[name_or_path]
    if os.path.isdir(name_or_path):
        return name_or_path
    raise ValueError(
        f"{name_or_path!r} is neither a bundled model "
        f"({sorted(CHECKPOINTS)}) nor an existing directory"
    )


def load_model(
    name_or_path: str = "IgLM",
    tokenizer: Optional[IgLMTokenizer] = None,
    device: Optional[torch.device] = None,
    gradient_checkpointing: bool = False,
) -> transformers.GPT2LMHeadModel:
    """Load an IgLM checkpoint with its token ids corrected, ready to train."""
    tokenizer = tokenizer or IgLMTokenizer()
    path = resolve_checkpoint(name_or_path)

    config = transformers.GPT2Config.from_pretrained(path)
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.cls_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model = transformers.GPT2LMHeadModel.from_pretrained(path, config=config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if device is not None:
        model = model.to(device)

    return model


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
