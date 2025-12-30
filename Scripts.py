import os
import sys
from typing import List
import sacrebleu
from reference_tokenization import STLTokenizer

def load_file(filename: str) -> List[str]:
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def load_references(ref_dir: str, num_refs: int = 1) -> List[List[str]]:
    references = []
    for i in range(num_refs):
        ref_file = os.path.join(ref_dir, f'ref.{i}.en')  # Assuming English references
        if os.path.exists(ref_file):
            refs = load_file(ref_file)
            if i == 0:
                references = [[r] for r in refs]  # Initialize as list of lists
            else:
                for j, ref in enumerate(refs):
                    references[j].append(ref)
        else:
            print(f"Warning: Reference file {ref_file} not found. Using single reference mode.")
            break
    return references
def evaluate_bleu(hypotheses: List[str], references: List[List[str]], tokenizer: STLTokenizer = None) -> dict:
    if tokenizer is None:
        tokenizer = STLTokenizer()

    # Tokenize references
    tokenized_refs = []
    for ref_list in references:
        tokenized_ref_list = []
        for ref in ref_list:
            tokens = tokenizer.tokenize(ref)
            tokenized_ref_list.append(' '.join(tokens))
        tokenized_refs.append(tokenized_ref_list)
    bleu = sacrebleu.corpus_bleu(
        sys=[hyp for hyp in hypotheses],  # Hypotheses (raw text; SacreBLEU will tokenize)
        ref=tokenized_refs,  # Tokenized references
        tokenize='13a',  # Moses tokenizer for hypotheses (standard for BLEU)
        smooth_method='exp(0.1)',  # Standard smoothing (method1 equivalent)
        force=False,  # Use default settings
        lowercase=True  # Case-insensitive BLEU
    )

    return {
        'bleu': bleu.score,
        'precisions': bleu.stats.precisions,
        'bp': bleu.stats.bp,
        'length_ratio': bleu.stats.sys_len / bleu.stats.ref_len,
        'full_score': str(bleu),  # Full SacreBLEU string for reproducibility
        'signature': bleu.format_signature()  # For exact reproducibility
    }
def main():
    # Configuration (adjust paths as needed)
    hypo_file = 'hypotheses.txt'  # Model outputs
    ref_dir = 'references/'  # Directory with reference files (e.g., ref.0.en)
    output_file = 'bleu_results.txt'

    # Load data
    if not os.path.exists(hypo_file):
        print(f"Error: Hypotheses file {hypo_file} not found.")
        return
    hypotheses = load_file(hypo_file)
    references = load_references(ref_dir)
    min_len = min(len(hypotheses), len(references))
    hypotheses = hypotheses[:min_len]
    references = references[:min_len]
    tokenizer = STLTokenizer()
    results = evaluate_bleu(hypotheses, references, tokenizer)
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"BLEU Score: {results['bleu']:.2f}\n")