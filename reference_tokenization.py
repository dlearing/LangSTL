import re
import sacrebleu
from typing import List, Tuple
import os
import nltk
from nltk.tokenize import word_tokenize

class STLTokenizer:
    def __init__(self):
        # Moses-style normalization rules
        self.normalization_pattern = re.compile(r'([^\s\.\!\?])([;:!?])')
        self.punctuation_pattern = re.compile(r'([^\.\!\?])\.(?=[^0-9A-Za-z])')
        self.closing_pattern = re.compile(r'([^\.\!\?])\)(?=[^0-9A-Za-z])')
        self.quote_pattern = re.compile(r'([^\.\!\?])"(?=[^0-9A-Za-z])')
        self.paren_pattern = re.compile(r'([^\.\!\?])\)(?=[^0-9A-Za-z])')
        self.csl_gloss_pattern = re.compile(r'([A-Z]+)\*(\d+)?')
        self.number_pattern = re.compile(r'(\d+)([^\d\s])')
    def normalize(self, text: str) -> str:
        text = text.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?')
        text = text.replace('；', ';').replace('：', ':').replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'").replace('（', '(').replace('）', ')')
        # Fix common punctuation issues
        text = self.normalization_pattern.sub(r'\1 \2', text)
        text = self.punctuation_pattern.sub(r'\1 .', text)
        text = self.closing_pattern.sub(r'\1 )', text)
        text = self.quote_pattern.sub(r'\1 "', text)
        text = self.paren_pattern.sub(r'\1 )', text)
        text = self.csl_gloss_pattern.sub(r'\1 *', text)
        text = self.number_pattern.sub(r'\1 \2', text)
        text = text.lower()
        return text.strip()
    def tokenize(self, text: str) -> List[str]:
        normalized = self.normalize(text)
        tokens = word_tokenize(normalized)
        processed_tokens = []
        for token in tokens:
            token = re.sub(r"n't\b", " not", token)
            token = re.sub(r"'s\b", " 's", token)
            token = re.sub(r"'re\b", " 're", token)
            token = re.sub(r"'ve\b", " 've", token)
            token = re.sub(r"'ll\b", " 'll", token)
            token = re.sub(r"'d\b", " 'd", token)
            # Handle hyphens
            token = re.sub(r'(\w+)(-)(\w+)', r'\1 - \3', token)
            if '*' in token:
                parts = token.split('*')
                processed_tokens.extend([p.strip() for p in parts if p.strip()])
            else:
                processed_tokens.append(token)
        return [t for t in processed_tokens if t]  # Remove empty tokens
