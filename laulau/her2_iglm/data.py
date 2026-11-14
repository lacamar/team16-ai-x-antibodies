"""Turn the Chinery et al. trastuzumab FACS table into IgLM infilling examples.

The dataset (``tz_her2_facs_3classes_524346.csv.zip``) has a very regular shape, which
we exploit and then assert:

  * ``vh_sequence`` is always 120 aa,
  * ``cdrh3`` is always 10 aa and always sits at ``vh[98:108]``,
  * the surrounding framework is byte-identical in every row,
  * ``vl_sequence`` is trastuzumab's, identical in every row.

So every training example is the same constant token template with 10 variable
positions. We therefore cache only the 10-residue spans plus one template row, and
build batches by writing the spans into a repeated template. That keeps the cache at
~5 MB instead of 65 MB and needs no padding or attention masking anywhere.

Token layout (length 125), matching ``iglm.model.utils.mask_span(..., append_span=True)``
so that a fine-tuned model stays compatible with ``IgLM.infill``:

    [HEAVY] [HUMAN] <vh[0:98]> [MASK] <vh[108:120]> [SEP] <cdrh3> [CLS]
    0       1        2..99      100    101..112      113    114..123 124

Loss is computed on positions 114..124 only -- the 10 span residues plus the final
[CLS] -- which is exactly the slice ``IgLM.log_likelihood`` scores for an infill range.
Teaching the model to emit [CLS] at the right place matters: ``IgLM._generate`` uses it
as the EOS token, so a model that forgets it never terminates.
"""

from __future__ import annotations

import dataclasses
import os
import zipfile
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from her2_iglm.tokenizer import IgLMTokenizer

#: Trastuzumab's own CDRH3 in this dataset's 10-residue numbering.
TRASTUZUMAB_CDRH3 = "WGGDGFYAMD"

#: 0-indexed, end-exclusive CDRH3 window inside the 120-aa VH.
CDRH3_RANGE: Tuple[int, int] = (98, 108)

VH_LENGTH = 120
CDRH3_LENGTH = CDRH3_RANGE[1] - CDRH3_RANGE[0]

#: Position of each field in the 125-token example.
MASK_POS = 2 + CDRH3_RANGE[0]
SEP_POS = MASK_POS + 1 + (VH_LENGTH - CDRH3_RANGE[1])
SPAN_START = SEP_POS + 1
SPAN_END = SPAN_START + CDRH3_LENGTH
CLS_POS = SPAN_END
SEQ_LENGTH = CLS_POS + 1

#: Loss targets are tokens SPAN_START..CLS_POS inclusive.
LOSS_FIRST_TARGET = SPAN_START
LOSS_LAST_TARGET = CLS_POS

CLASS_NAMES = ("low", "medium", "high")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclasses.dataclass
class Splits:
    """Pre-tokenised CDRH3 spans and binding classes, split three ways."""

    template: np.ndarray  # (125,) int64, the constant token row
    train_spans: np.ndarray  # (n, 10) uint8 token ids
    train_classes: np.ndarray  # (n,) uint8, index into CLASS_NAMES
    val_spans: np.ndarray
    val_classes: np.ndarray
    test_spans: np.ndarray
    test_classes: np.ndarray
    framework_prefix: str
    framework_suffix: str
    vl_sequence: str
    dropped_rows: int

    def counts(self) -> Dict[str, Dict[str, int]]:
        out = {}
        for name, classes in (
            ("train", self.train_classes),
            ("val", self.val_classes),
            ("test", self.test_classes),
        ):
            out[name] = {
                CLASS_NAMES[idx]: int((classes == idx).sum())
                for idx in range(len(CLASS_NAMES))
            }
            out[name]["total"] = int(classes.size)
        return out


def _read_facs_csv(path: str) -> pd.DataFrame:
    """Read the FACS table, transparently handling the .zip wrapper."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            # Skip macOS resource forks (__MACOSX/._*), which are not real CSVs.
            members = [
                n
                for n in archive.namelist()
                if n.endswith(".csv") and not os.path.basename(n).startswith("._")
            ]
            if len(members) != 1:
                raise ValueError(f"expected exactly one CSV in {path}, got {members}")
            with archive.open(members[0]) as handle:
                return pd.read_csv(handle)
    return pd.read_csv(path)


def build_splits(
    csv_path: str,
    tokenizer: IgLMTokenizer,
    val_size: int = 8000,
    test_size: int = 8000,
    seed: int = 0,
    species_token: str = "[HUMAN]",
    chain_token: str = "[HEAVY]",
) -> Splits:
    """Load, validate and split the FACS table into infilling examples."""
    frame = _read_facs_csv(csv_path)

    required = {"cdrh3", "binding_class", "vh_sequence", "vl_sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")

    n_raw = len(frame)

    # Keep only rows with the canonical geometry. Anything else would quietly
    # corrupt the fixed-template assumption, so drop it loudly instead.
    start, end = CDRH3_RANGE
    ok = (
        frame["vh_sequence"].str.len().eq(VH_LENGTH)
        & frame["cdrh3"].str.len().eq(CDRH3_LENGTH)
        & frame["binding_class"].isin(CLASS_NAMES)
        & (frame["vh_sequence"].str.slice(start, end) == frame["cdrh3"])
    )
    frame = frame.loc[ok].reset_index(drop=True)

    prefixes = frame["vh_sequence"].str.slice(0, start)
    suffixes = frame["vh_sequence"].str.slice(end, VH_LENGTH)
    framework_prefix = prefixes.mode().iat[0]
    framework_suffix = suffixes.mode().iat[0]

    same_framework = prefixes.eq(framework_prefix) & suffixes.eq(framework_suffix)
    frame = frame.loc[same_framework].reset_index(drop=True)

    dropped = n_raw - len(frame)

    vl_sequence = frame["vl_sequence"].mode().iat[0]

    # --- build the constant template row -------------------------------------
    template = np.zeros(SEQ_LENGTH, dtype=np.int64)
    template[0] = tokenizer.vocab[chain_token]
    template[1] = tokenizer.vocab[species_token]
    template[2:MASK_POS] = tokenizer.encode_residues(framework_prefix)
    template[MASK_POS] = tokenizer.mask_token_id
    template[MASK_POS + 1 : SEP_POS] = tokenizer.encode_residues(framework_suffix)
    template[SEP_POS] = tokenizer.sep_token_id
    template[SPAN_START:SPAN_END] = tokenizer.unk_token_id  # overwritten per row
    template[CLS_POS] = tokenizer.cls_token_id

    # --- vectorised span encoding -------------------------------------------
    # Build a 256-entry byte -> token-id table, then index into it. This turns
    # 524k x 10 lookups into one numpy gather instead of a Python loop.
    lookup = np.full(256, tokenizer.unk_token_id, dtype=np.uint8)
    for residue, token_id in tokenizer.vocab.items():
        if len(residue) == 1:
            lookup[ord(residue)] = token_id

    raw = frame["cdrh3"].to_numpy(dtype="U%d" % CDRH3_LENGTH)
    span_bytes = np.frombuffer(
        "".join(raw.tolist()).encode("ascii"), dtype=np.uint8
    ).reshape(-1, CDRH3_LENGTH)
    spans = lookup[span_bytes]

    if (spans == tokenizer.unk_token_id).any():
        bad = int((spans == tokenizer.unk_token_id).any(axis=1).sum())
        raise ValueError(f"{bad} CDRH3 sequences contain non-standard residues")

    classes = (
        frame["binding_class"].map(CLASS_TO_INDEX).to_numpy(dtype=np.uint8)
    )

    # --- stratified split ----------------------------------------------------
    rng = np.random.default_rng(seed)
    val_idx, test_idx = [], []
    for class_index in range(len(CLASS_NAMES)):
        pool = np.flatnonzero(classes == class_index)
        rng.shuffle(pool)
        share = pool.size / classes.size
        n_val = int(round(val_size * share))
        n_test = int(round(test_size * share))
        val_idx.append(pool[:n_val])
        test_idx.append(pool[n_val : n_val + n_test])

    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    held_out = np.zeros(classes.size, dtype=bool)
    held_out[val_idx] = True
    held_out[test_idx] = True
    train_idx = np.flatnonzero(~held_out)
    rng.shuffle(train_idx)

    return Splits(
        template=template,
        train_spans=spans[train_idx],
        train_classes=classes[train_idx],
        val_spans=spans[val_idx],
        val_classes=classes[val_idx],
        test_spans=spans[test_idx],
        test_classes=classes[test_idx],
        framework_prefix=framework_prefix,
        framework_suffix=framework_suffix,
        vl_sequence=vl_sequence,
        dropped_rows=dropped,
    )


def save_splits(splits: Splits, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        template=splits.template,
        train_spans=splits.train_spans,
        train_classes=splits.train_classes,
        val_spans=splits.val_spans,
        val_classes=splits.val_classes,
        test_spans=splits.test_spans,
        test_classes=splits.test_classes,
        framework_prefix=np.array(splits.framework_prefix),
        framework_suffix=np.array(splits.framework_suffix),
        vl_sequence=np.array(splits.vl_sequence),
        dropped_rows=np.array(splits.dropped_rows),
    )


def load_splits(path: str) -> Splits:
    with np.load(path, allow_pickle=False) as blob:
        return Splits(
            template=blob["template"],
            train_spans=blob["train_spans"],
            train_classes=blob["train_classes"],
            val_spans=blob["val_spans"],
            val_classes=blob["val_classes"],
            test_spans=blob["test_spans"],
            test_classes=blob["test_classes"],
            framework_prefix=str(blob["framework_prefix"]),
            framework_suffix=str(blob["framework_suffix"]),
            vl_sequence=str(blob["vl_sequence"]),
            dropped_rows=int(blob["dropped_rows"]),
        )


def prepare(
    csv_path: str,
    cache_path: str,
    tokenizer: Optional[IgLMTokenizer] = None,
    force: bool = False,
    **kwargs,
) -> Splits:
    """Return cached splits, building them from the CSV on first use."""
    if os.path.exists(cache_path) and not force:
        return load_splits(cache_path)
    splits = build_splits(csv_path, tokenizer or IgLMTokenizer(), **kwargs)
    save_splits(splits, cache_path)
    return splits
