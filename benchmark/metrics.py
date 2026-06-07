#!/usr/bin/env python3
"""
benchmark/metrics.py

Metrics module for the benchmarking suite. Includes ROUGE, BLEU, BERTScore,
and Task Accuracy calculations.
"""

import re
import math
import numpy as np
import torch
from collections import Counter

def tokenize_text(text):
    if not isinstance(text, str):
        if text is None or (isinstance(text, float) and math.isnan(text)):
            text = ""
        else:
            text = str(text)
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens

def calculate_rouge_1(candidate, reference):
    cand_tokens = tokenize_text(candidate)
    ref_tokens = tokenize_text(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    cand_set = set(cand_tokens)
    ref_set = set(ref_tokens)
    overlap = len(cand_set.intersection(ref_set))
    precision = overlap / len(cand_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def calculate_rouge_l(candidate, reference):
    cand_tokens = tokenize_text(candidate)
    ref_tokens = tokenize_text(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    m = len(cand_tokens)
    n = len(ref_tokens)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif cand_tokens[i - 1] == ref_tokens[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    lcs_len = L[m][n]
    precision = lcs_len / m
    recall = lcs_len / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def calculate_bleu_1(candidate, reference):
    cand_tokens = tokenize_text(candidate)
    ref_tokens = tokenize_text(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    cand_counts = {}
    for t in cand_tokens:
        cand_counts[t] = cand_counts.get(t, 0) + 1
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    matches = sum(min(count, ref_counts.get(t, 0)) for t, count in cand_counts.items())
    precision = matches / len(cand_tokens)
    c = len(cand_tokens)
    r = len(ref_tokens)
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c) if c > 0 else 0.0
    return bp * precision

def calculate_bleu_4(candidate, reference):
    cand_tokens = tokenize_text(candidate)
    ref_tokens = tokenize_text(reference)
    c = len(cand_tokens)
    r = len(ref_tokens)
    if c == 0 or r == 0:
        return 0.0
    precisions = []
    for n in range(1, 5):
        if c < n or r < n:
            precisions.append(0.0)
            continue
        cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(c - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(r - n + 1)]
        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)
        matches = sum(min(count, ref_counts[ngram]) for ngram, count in cand_counts.items())
        precision = (matches + 1) / (len(cand_ngrams) + 1)
        precisions.append(precision)
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)
    try:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    except ValueError:
        geo_mean = 0.0
    return bp * geo_mean

def calculate_exact_match(candidate, reference):
    cand_clean = " ".join(tokenize_text(candidate))
    ref_clean = " ".join(tokenize_text(reference))
    return 1.0 if cand_clean == ref_clean else 0.0

def calculate_bertscore(candidate, reference, model_name="bert-base-uncased"):
    global _bertscore_model, _bertscore_tokenizer, _bertscore_failed
    if globals().get('_bertscore_failed', False):
        return _bertscore_fallback(candidate, reference)

    try:
        if '_bertscore_model' not in globals():
            from transformers import AutoTokenizer, AutoModel
            _bertscore_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            _bertscore_model = AutoModel.from_pretrained(model_name, local_files_only=False)
            _bertscore_model.eval()
            if torch.cuda.is_available():
                _bertscore_model = _bertscore_model.to('cuda')
        
        device = next(_bertscore_model.parameters()).device
        with torch.no_grad():
            ref_tokens = _bertscore_tokenizer(reference, return_tensors="pt", padding=False, truncation=True)
            cand_tokens = _bertscore_tokenizer(candidate, return_tensors="pt", padding=False, truncation=True)
            ref_tokens = {k: v.to(device) for k, v in ref_tokens.items()}
            cand_tokens = {k: v.to(device) for k, v in cand_tokens.items()}
            
            ref_outputs = _bertscore_model(**ref_tokens)
            cand_outputs = _bertscore_model(**cand_tokens)
            ref_emb = ref_outputs.last_hidden_state[0]
            cand_emb = cand_outputs.last_hidden_state[0]
            ref_emb_norm = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
            cand_emb_norm = cand_emb / cand_emb.norm(dim=-1, keepdim=True)
            sim_matrix = torch.matmul(cand_emb_norm, ref_emb_norm.t())
            precision = sim_matrix.max(dim=-1)[0].mean().item()
            recall = sim_matrix.max(dim=0)[0].mean().item()
            if precision + recall == 0:
                return 0.0
            f1 = 2 * precision * recall / (precision + recall)
            return float(f1)
    except Exception:
        _bertscore_failed = True
        return _bertscore_fallback(candidate, reference)

def _bertscore_fallback(candidate, reference):
    r1 = calculate_rouge_1(candidate, reference)
    rl = calculate_rouge_l(candidate, reference)
    b1 = calculate_bleu_1(candidate, reference)
    b4 = calculate_bleu_4(candidate, reference)
    return float((r1 + rl + b1 + b4) / 4.0)

def calculate_task_accuracy(candidate, reference):
    if not isinstance(candidate, str):
        candidate = str(candidate) if candidate is not None else ""
    if not isinstance(reference, str):
        reference = str(reference) if reference is not None else ""
    ref_match = re.match(r'^\s*\[([^\]]+)\]', reference)
    if not ref_match:
        return calculate_exact_match(candidate, reference)
    ref_prefix = ref_match.group(1).strip().upper()
    cand_match = re.search(r'\[([^\]]+)\]', candidate)
    if not cand_match:
        clean_cand = candidate.strip().upper()
        if ref_prefix in clean_cand[:len(ref_prefix)+10]:
            return 1.0
        return 0.0
    cand_prefix = cand_match.group(1).strip().upper()
    def normalize_prefix(p):
        return p.replace("AND", "&").replace(" ", "")
    return 1.0 if normalize_prefix(ref_prefix) == normalize_prefix(cand_prefix) else 0.0

_semantic_model = None

def calculate_semantic_similarity(candidate, reference):
    global _semantic_model

    try:
        if _semantic_model is None:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

        emb1 = _semantic_model.encode(candidate)
        emb2 = _semantic_model.encode(reference)

        import numpy as np

        score = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) *
            np.linalg.norm(emb2)
        )

        return float(score)

    except Exception:
        return calculate_bertscore(candidate, reference)

def calculate_gpt_judge_score(candidate, reference, prompt, api_url, model_name, api_key):
    import json
    import urllib.request
    import re
    
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        return 0.0
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional educational curriculum grader assessing Bloom's Taxonomy alignment. Return ONLY a single integer score between 1 and 5."
            },
            {
                "role": "user",
                "content": (
                    f"Passage Context:\n{prompt}\n\n"
                    f"Reference Question:\n{reference}\n\n"
                    f"Candidate Question:\n{candidate}\n\n"
                    "Assess if the Candidate Question is grammatically correct, matches the educational scope of the Passage, "
                    "and targets the same Bloom's Taxonomy level as the Reference Question. "
                    "Provide your rating as a single digit integer from 1 (poor/incorrect) to 5 (excellent/perfect alignment)."
                )
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1
    }
    
    try:
        req = urllib.request.Request(api_url, data=json.dumps(payload).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode('utf-8'))
            score_str = res['choices'][0]['message']['content'].strip()
            match = re.search(r'[1-5]', score_str)
            if match:
                return float(match.group(0))
    except Exception:
        pass
    return 0.0

