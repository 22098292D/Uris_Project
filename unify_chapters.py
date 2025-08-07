#!/usr/bin/env python3
# unify_chapters.py
#python slides2kgs-topic-tree\unify_chapters.py --chapters_dir "course_resultsV2\course_slides\chapters" --output_dir "course_resultsV2\course_slides\unified"
import os
import json
import glob
import argparse

def unify_chapters(chapters_dir: str, output_dir: str):
    """
    把 chapters_dir 下所有子目录的 gpt_topic_hierarchy.json 统一合并到一个根节点里。
    每个章节节点都保持原有的子树结构，只把 type 统一标为 'category'。
    """
    # 找到所有章节 JSON
    pattern = os.path.join(chapters_dir, '*', 'gpt_topic_hierarchy.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"未找到任何 gpt_topic_hierarchy.json（路径：{pattern}）")
        return

    children = []
    total_slides = 0
    for fn in files:
        hierarchy = json.load(open(fn, encoding='utf-8'))
        # 将每个章节的根节点当作一个 category
        hierarchy['type'] = 'category'
        # 累计 slide 数
        total_slides += hierarchy.get('slide_count', 0)
        children.append(hierarchy)

    course_name = os.path.basename(os.path.dirname(chapters_dir))
    unified = {
        'type': 'root',
        'title': f"{course_name} - Complete Course",
        'children': children,
        'slide_count': total_slides
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'course_topic_hierarchy.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    print(f"✅ 已生成统一文件：{out_path}")
    print(f"   - 共 {len(children)} 个章节，{total_slides} 张幻灯片")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unify per-chapter GPT hierarchies into one")
    p.add_argument("--chapters_dir", required=True, help="目录，形如 ./course_results/chapters")
    p.add_argument("--output_dir", required=True, help="生成 unified JSON 的目标文件夹")
    args = p.parse_args()
    unify_chapters(args.chapters_dir, args.output_dir)
