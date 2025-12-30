import os
from typing import List
from sacrebleu.metrics import BLEU, CHRF, TER
from sacremoses import MosesTokenizer

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
def evaluate_bleu(target: List[str], references: List[List[str]], tokenizer: MosesTokenizer = None) -> dict:
    if tokenizer is None:
        tokenizer = MosesTokenizer()

    # Tokenize references
    tokenized_refs = []
    for ref_list in references:
        tokenized_ref_list = []
        for ref in ref_list:
            tokens = tokenizer.tokenize(ref)
            tokenized_ref_list.append(' '.join(tokens))
        tokenized_refs.append(tokenized_ref_list)
    bleu = BLEU(
        lowercase=True,
        tokenize='13a',
        smooth_method='exp',
        smooth_value=0.1,
        force=False,
        effective_order=True
    )
    bleu.corpus_score(target, tokenized_refs)
    return {
        'bleu': bleu.score,
        'signature': bleu.get_signature()
    }
def main():
    hypo_file = 'target.txt'
    ref_dir = 'references/'
    output_file = 'bleu_results.txt'

    # Load data
    if not os.path.exists(hypo_file):
        print(f"Error: target file {hypo_file} not found.")
        return
    target = load_file(hypo_file)
    references = load_references(ref_dir)
    min_len = min(len(target), len(references))
    hypotheses = target[:min_len]
    references = references[:min_len]
    tokenizer = MosesTokenizer()
    results = evaluate_bleu(target, references, tokenizer)
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"BLEU Score: {results['bleu']:.2f}\n")