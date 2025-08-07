#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import openai

# ———— 参数区 ————
OPENAI_MODEL       = "gpt-4o-mini"
SBERT_MODEL        = "all-MiniLM-L6-v2"
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
FUZZY_THRESHOLD    = 80     # 提高模糊匹配阈值
SEMANTIC_THRESHOLD = 0.2     # 提高语义匹配阈值
HEAD_FUZZY_THRESH  = 80     # 标题模糊匹配阈值
GPT_MAX_CALLS      = 200     # 限制 GPT 调用次数
# ————————————

openai.api_key = OPENAI_API_KEY

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[\(\)\[\]\,\.\;\:\-\—\"\'\?]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def advanced_normalize(text: str) -> str:
    text = normalize(text)
    # 去掉常见后缀和停用词
    for suf in [r'\btechniques?\b', r'\bmethods?\b', r'\bsystems?\b',
                r'\bapproaches?\b', r'\baspects?\b', r'\bfoundations?\b']:
        text = re.sub(suf, '', text)
    text = re.sub(r'\b(of|and|in|for|the|a|an)\b', '', text)
    text = re.sub(r'(\w+)s\b', r'\1', text)  # 简单复数到单数
    return re.sub(r'\s+', ' ', text).strip()

def load_and_prep(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)[['Head','Tail']]
    df['Head'] = df['Head'].map(advanced_normalize)
    df['Tail'] = df['Tail'].map(advanced_normalize)
    return df.drop_duplicates().reset_index(drop=True)

def build_map(df: pd.DataFrame) -> dict:
    m = {}
    for head, tail in df.itertuples(index=False):
        m.setdefault(head, []).append(tail)
    return m

def find_top_k_similar(item: str, candidates: list, k: int, sbert) -> list:
    """
    返回对 item 最相似的至多 k 个 candidates（相似度 >= SEMANTIC_THRESHOLD）。
    """
    if not candidates:
        return []
    emb_cand = sbert.encode(candidates, convert_to_numpy=True)
    emb_item = sbert.encode([item], convert_to_numpy=True)
    # L2 normalize
    emb_cand /= np.linalg.norm(emb_cand, axis=1, keepdims=True)
    emb_item /= np.linalg.norm(emb_item, axis=1, keepdims=True)
    k_use = min(k, len(candidates))
    nn = NearestNeighbors(n_neighbors=k_use, metric='cosine').fit(emb_cand)
    distances, indices = nn.kneighbors(emb_item)
    sim = 1 - distances.flatten()
    result = []
    for idx, s in zip(indices.flatten(), sim):
        if s >= SEMANTIC_THRESHOLD:
            result.append(candidates[idx])
    return result

def gpt_equivalent_or_contains(a: str, b: str) -> bool:
    prompt = (
        f"概念A: “{a}”\n"
        f"概念B: “{b}”\n"
        "问：这两个概念表示的是同一主题或种类吗？或者一个是另一个的子类/包含关系？你需要尽量放宽你的判定范围\n"
        "请只回答“是”或“否”。"
    )
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":"你是一位学术领域知识图谱专家。"},
            {"role":"user","content":prompt}
        ],
        max_tokens=10,
        temperature=0.0
    )
    ans = resp.choices[0].message.content.strip()
    print(ans)
    return ans.startswith("是")

def semantic_match(pred_list, gt_list, sbert):
    emb_pr = sbert.encode(pred_list, convert_to_numpy=True)
    emb_gt = sbert.encode(gt_list, convert_to_numpy=True)
    emb_pr /= np.linalg.norm(emb_pr, axis=1, keepdims=True)
    emb_gt /= np.linalg.norm(emb_gt, axis=1, keepdims=True)
    nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(emb_gt)
    distances, indices = nn.kneighbors(emb_pr)
    sim = 1 - distances.flatten()
    matches = []
    for i, p in enumerate(pred_list):
        if sim[i] >= SEMANTIC_THRESHOLD:
            matches.append((p, gt_list[indices[i,0]]))
    return matches

def match_for_head(gt_tails, pred_tails, sbert):
    gt = set(gt_tails)
    pr = set(pred_tails)
    tp = set()

    # 1. Exact
    for p in list(pr):
        if p in gt:
            tp.add(p)
            pr.discard(p)
            gt.discard(p)

    # 2. Substring
    for p in list(pr):
        for g in list(gt):
            if p in g or g in p:
                tp.add(p)
                pr.discard(p)
                gt.discard(g)
                break

    # 3. Fuzzy
    for p in list(pr):
        best, score = None, 0
        for g in gt:
            sc = fuzz.token_set_ratio(p, g)
            if sc > score:
                best, score = g, sc
        if score >= FUZZY_THRESHOLD:
            tp.add(p)
            pr.discard(p)
            gt.discard(best)

    # 4. Semantic (kNN via sklearn)
    if sbert and pr and gt:
        pred_list, gt_list = list(pr), list(gt)
        for p, g in semantic_match(pred_list, gt_list, sbert):
            tp.add(p)
            pr.discard(p)
            gt.discard(g)

    # 5. GPT Check
    if gt:
        print(f"[DEBUG] 进入 GPT 检查，pr={pr}, gt={gt}", flush=True)
        calls = 0
        for p in list(pr):
            if calls >= GPT_MAX_CALLS:
                break
            # find_top_k_similar 已经保证不会超过 gt 的大小
            for g in find_top_k_similar(p, list(gt), k=3, sbert=sbert):
                ans = gpt_equivalent_or_contains(p, g)
                print(f"[DEBUG GPT] 判断 {p} vs {g} => {ans}", flush=True)
                if ans:
                    tp.add(p)
                    pr.discard(p)
                    gt.discard(g)
                    calls += 1
                    break

    return tp, pr, gt


def map_pred_heads_to_gt(gt_heads, pred_heads, sbert):
    mapping = {}
    # Exact
    for p in pred_heads:
        if p in gt_heads:
            mapping[p] = p
    # Fuzzy
    for p in pred_heads:
        if p in mapping: continue
        best, score = None, 0
        for g in gt_heads:
            sc = fuzz.token_set_ratio(p, g)
            if sc > score:
                best, score = g, sc
        if score >= HEAD_FUZZY_THRESH:
            mapping[p] = best
    # Semantic
    rem = [p for p in pred_heads if p not in mapping]
    if rem:
        for p, g in semantic_match(rem, gt_heads, sbert):
            mapping[p] = g
    # GPT verify
    calls = 0
    for p in pred_heads:
        if p in mapping or calls >= GPT_MAX_CALLS: continue
        for g in find_top_k_similar(p, gt_heads, k=3, sbert=sbert):
            if gpt_equivalent_or_contains(p, g):
                mapping[p] = g
                calls += 1
                break
    return mapping

if __name__ == "__main__":
    # 1. 加载
    #ground truth 
    df_gt   = load_and_prep("nodes\Knowledge Graph.xlsx")
    #预测结果
    df_pred = load_and_prep("course_resultsV2\course_slides\extractedV2.xlsx")

    # 2. Head 映射
    sbert = SentenceTransformer(SBERT_MODEL)
    gt_heads   = sorted(df_gt['Head'].unique())
    pred_heads = sorted(df_pred['Head'].unique())
    head_map   = map_pred_heads_to_gt(gt_heads, pred_heads, sbert)

    # 3. 重建 pred_map
    pred_map = {}
    for h, t in df_pred.itertuples(index=False):
        mapped = head_map.get(h)
        if not mapped: continue
        pred_map.setdefault(mapped, []).append(t)
    gt_map = build_map(df_gt)

    # 4. 匹配
    per_head = {}
    for head, gt_tails in gt_map.items():
        pred_tails = pred_map.get(head, [])
        tp, fp, fn = match_for_head(gt_tails, pred_tails, sbert)
        per_head[head] = {'tp':tp, 'fp':fp, 'fn':fn}

    # 5. 统计
    TP = sum(len(v['tp']) for v in per_head.values())
    FP = sum(len(v['fp']) for v in per_head.values())
    FN = sum(len(v['fn']) for v in per_head.values())
    prec = TP/(TP+FP) if TP+FP else 0
    rec  = TP/(TP+FN) if TP+FN else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0

    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1-score:  {f1:.2%}")
