#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd

# 根目录：存放各课程子文件夹
ROOT_DIR = "course_resultsV2"

def extract_pairs_from_node(head_category, node, pairs):
    """
    递归提取：以 head_category 为父概念，
    对 node["children"] 中的每个子节点，
    用它们的 category（优先）或 key 作为 tail，
    并继续向下钻取。
    跳过任何 child_name == "leaf" 的节点。
    """
    if not isinstance(node, dict):
        return

    children = node.get("children")
    if not children:
        return

    # 展平 children
    entries = []
    if isinstance(children, dict):
        entries = list(children.items())
    else:
        for elem in children:
            if isinstance(elem, dict):
                entries.extend(elem.items())

    for child_name, child_node in entries:
        # 先跳过键名为 "leaf" 的子节点
        if child_name == "leaf":
            continue
        if not isinstance(child_node, dict):
            continue

        # 取概念名：优先 category，否则用 key
        cat = child_node.get("category")
        tail = cat.strip() if isinstance(cat, str) and cat.strip() else child_name

        pairs.append((head_category, "Contain Subtopic", tail))
        extract_pairs_from_node(tail, child_node, pairs)


def process_json(json_path):
    """
    1. 读取 JSON，取 data["title"] 作为 root_title；
    2. 展平 data["children"] 列表中所有 dict 或 list 元素，收集第一层 (key, node)；
    3. 对每个第一层节点：
         - 跳过 name == "leaf"；
         - 添加 root_title -> tail（三元组），
         - 递归 extract_pairs_from_node。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    root_title = data.get("title", "").strip()
    all_children = data.get("children", [])
    if not root_title or not all_children:
        return []

    # 收集所有第一层 (name, node)
    root_items = []
    for container in all_children:
        if isinstance(container, dict):
            root_items.extend(container.items())
        elif isinstance(container, list):
            for elem in container:
                if isinstance(elem, dict):
                    root_items.extend(elem.items())

    pairs = []
    for name, node in root_items:
        # 跳过所有 name == "leaf"
        if name == "leaf":
            continue
        if not isinstance(node, dict):
            continue

        # 取概念名：优先 category，否则用 key
        cat = node.get("category")
        tail = cat.strip() if isinstance(cat, str) and cat.strip() else name

        pairs.append((root_title, "Contain Subtopic", tail))
        extract_pairs_from_node(tail, node, pairs)

    return pairs









def main():
    for course in os.listdir(ROOT_DIR):
        course_folder = os.path.join(ROOT_DIR, course)
        if not os.path.isdir(course_folder):
            continue

        json_file = os.path.join(course_folder, "unified", "course_topic_hierarchy.json")
        if not os.path.isfile(json_file):
            print(f"跳过（未找到 JSON）：{course_folder}")
            continue

        print(f"正在处理：{json_file}")
        triples = process_json(json_file)
        if not triples:
            print(f"  警告：未提取到任何概念对")
            continue

        df = pd.DataFrame(triples, columns=["Head", "Relation", "Tail"])
        excel_path = os.path.join(course_folder, "extractedV2.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"  已生成：{excel_path}")

if __name__ == "__main__":
    main()
