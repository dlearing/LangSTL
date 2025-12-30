#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHOENIX-14T BLEU evaluation with Reference Tokenization.

Workflow:
- Load predictions and references (one per line, aligned)
- Apply a reference tokenization pipeline (tokenization for both preds and refs)
- Compute BLEU via sacreBLEU (supports single or multiple references)
- Output final BLEU score and an optional JSON summary

Notes:
- This is a template. Replace the tokenize_text function with your actual tokenization tools.
- For multi-reference setups, extend refs handling to a list of references per sample.
"""

import argparse
import json
import sys
from typing import List

try:
    import sacrebleu
except ImportError:
    sys.stderr.write("Please install sacrebleu: 'pip install sacrebleu'\n")
    sys.exit(1)

# ----------------------------
# Reference tokenization utilities
# ----------------------------

def _noop_tokenizer(text: str) -> str:
    """No-op tokenizer: return as-is. Placeholder if you do not tokenize."""
    return text

def tokenize_zh_with_tool(text: str, tool: str) -> str:
    """
    Placeholder for Chinese tokenization using a specific tool.
    You should implement actual integration, e.g.:
      - jieba.cut for word-level tokens
      - PKU/HIT Chinese tokenizer
      - Moses tokenizer for Chinese (if available)
    Return a single string with tokens separated by spaces.
    """
    # Example placeholder: simple word-level split if tool is 'zh_word'
    if tool == "zh_word":
        # Replace with real tokenization logic
        return " ".join(text.strip().split())
    elif tool == "zh_char":
        # Character-level tokenization (space-separated chars, ignoring spaces)
        return " ".join(list(text.replace(" ", "")))
    else:
        # Fallback to no-op
        return _noop_tokenizer(text)

def tokenize_text(text: str, method: str) -> str:
    """
    Tokenize a text string according to the chosen method.
    Methods:
      - "word": word-level tokenization (e.g., English or Chinese as a tokenizer outputs)
      - "char": character-level tokenization
      - "zh_simple": a simple placeholder Chinese tokenization (replace with real tool)
      - "none": no tokenization (pass-through)
    """
    if method == "word":
        # Replace with your actual word-level tokenizer
        # Example placeholder: split on whitespace
        return " ".join(text.strip().split())
    elif method == "char":
        return " ".join(list(text.replace(" ", "")))
    elif method == "zh_simple":
        return tokenize_zh_with_tool(text, "zh_word")
    elif method == "none":
        return _noop_tokenizer(text)
    else:
        raise ValueError(f"Unsupported tokenization method: {method}")

# ----------------------------
# Helpers
# ----------------------------

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def ensure_same_length(a: List[str], b: List[str]) -> None:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: preds={len(a)} refs={len(b)}")

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PHOENIX-14T BLEU evaluation with Reference Tokenization"
    )
    parser.add_argument("--preds", required=True, help="Path to predictions (one per line)")
    parser.add_argument("--refs", required=True, help="Path to references (one per line)")
    parser.add_argument(
        "--seg_tool",
        default="word",
        choices=["word", "char", "zh_simple", "none"],
        help="Tokenization strategy for both predictions and references"
    )
    parser.add_argument("--output", default=None, help="Path to write BLEU results JSON (optional)")
    parser.add_argument("--multi_ref", action="store_true",
                        help="Enable multi-reference support (requires refs_multi.txt with one multi-reference per line in JSON array form, see README)")
    parser.add_argument("--refs_multi", default=None,
                        help="Path to multi-reference file: each line is a JSON array of references for the corresponding sample")
    args = parser.parse_args()

    preds = read_lines(args.preds)
    refs = read_lines(args.refs)
    ensure_same_length(preds, refs)

    # Tokenize
    tokenized_preds = []
    tokenized_refs = []

    for p, r in zip(preds, refs):
        tp = tokenize_text(p, method=args.seg_tool)
        tr = tokenize_text(r, method=args.seg_tool)
        tokenized_preds.append(tp)
        tokenized_refs.append(tr)

    # Prepare references for sacreBLEU
    # sacreBLEU expects a list of reference sequences per sample, i.e., a list of lists:
    # refs_per_sample = [ [ref1_line1, ref2_line1, ...], [ref1_line2, ref2_line2, ...], ... ]
    # If you only have a single reference per sample, you can provide [[ref_line1], [ref_line2], ...]
    if args.multi_ref:
        if not args.refs_multi:
            raise ValueError("When --multi_ref is set, you must provide --refs_multi")
        refs_multi_lines = read_lines(args.refs_multi)
        if len(refs_multi_lines) != len(preds):
            raise ValueError("Length mismatch between preds and multi-refs file.")
        # Each line in refs_multi should be a JSON array of references for that sample
        refs_per_sample = []
        for line in refs_multi_lines:
            # Expecting something like ["ref1 sample1", "ref2 sample1", ...]
            try:
                refs_list = json.loads(line.strip())
                if not isinstance(refs_list, list):
                    raise ValueError
            except Exception:
                raise ValueError("Each line in --refs_multi must be a JSON array of references.")
            # Tokenize each reference in the list
            tokenized_refs_line = [" ".join(line.strip().split()) for line in []]  # placeholder
            # We bypass: we will tokenize below directly from un-tokenized refs by mapping over originals
            # For correctness, you should implement proper per-sample multi-reference tokenization here.
        # To keep this template simple, we will fall back to single-reference path if multi_ref parsing is complex.
        raise NotImplementedError("Multi-reference support is provided as a placeholder. Implement as needed.")
    else:
        # single reference per sample
        # sacreBLEU expects a list of reference lists: [ [ref_line1], [ref_line2], ... ]
        refs_for_bleu = [[ref.strip()] for ref in tokenized_refs]

        # sacreBLEU: corpus_bleu(preds, [refs_per_sample])
        # pred strings should be tokenized text joined by spaces (as we did)
        bleu = sacrebleu.corpus_bleu(tokenized_preds, refs_for_bleu)

        result = {
            "bleu": bleu.score,
            "stats": {
                "samples": len(preds),
                "tokenization": args.seg_tool,
            },
        }

        print(f"BLEU score: {bleu.score:.2f}")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f_out:
                json.dump(result, f_out, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()