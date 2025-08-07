#!/usr/bin/env python3
"""
Recursively process multiple course folders under a root directory,
extract slides, summarize titles semantically, organize with GPT,
and save per-course & unified outputs for each subfolder.
"""
import os
import json
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any

from utils.slide_preprocessor import SlideExtractor
from utils.gpt_topic_organizer import GPTTopicOrganizer
from utils.hierarchy_visualizer import HierarchyVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourseProcessor:
    def __init__(self, course_dir: str, output_dir: str = "course_results", force_reprocess: bool = False):
        self.course_dir = course_dir
        self.output_dir = output_dir
        self.force_reprocess = force_reprocess
        self.extractor = SlideExtractor()
        self.organizer = GPTTopicOrganizer()
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "chapters"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "unified"), exist_ok=True)

    def find_pdf_files(self) -> List[str]:
        """只查找本课程文件夹下的第一层 PDF"""
        pdfs = sorted(glob.glob(os.path.join(self.course_dir, "*.pdf")))
        logger.info(f"[{Path(self.course_dir).name}] Found {len(pdfs)} PDF files.")
        return pdfs

    def process_single_chapter(self, pdf_path: str) -> Dict[str, Any]:
        chapter = Path(pdf_path).stem
        chap_out = os.path.join(self.output_dir, "chapters", chapter)
        os.makedirs(chap_out, exist_ok=True)

        # Step 1: 提取幻灯片
        slides = self.extractor.extract_slides(pdf_path)
        with open(os.path.join(chap_out, "extracted_slides.json"), "w", encoding="utf-8") as f:
            json.dump(slides, f, indent=2, ensure_ascii=False)

        # Step 1.5: 语义总结每页标题
        logger.info(f"[{chapter}] Summarizing slide titles...")
        slides = self.organizer.summarize_slides(slides)

        # Step 2: GPT 组织主题
        hierarchy = self.organizer.organize_topics(slides)
        # 如果 root title 没被 GPT 覆写，就用目录名
        if hierarchy.get('title', '').lower() in ('presentation topics', ''):
            hierarchy['title'] = chapter
        else:
            hierarchy['title'] = f"{chapter}: {hierarchy['title']}"

        with open(os.path.join(chap_out, "gpt_topic_hierarchy.json"), "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)

        # Step 3: 可视化
        viz = HierarchyVisualizer()
        viz.save_visualization(hierarchy, chap_out, f"{chapter}_")

        return {
            'chapter_name': chapter,
            'slides_count': len(slides),
            'hierarchy': hierarchy,
            'output_dir': chap_out
        }

    def aggregate_chapters(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful = [r for r in results if 'hierarchy' in r]
        total_slides = sum(r['slides_count'] for r in successful)
        course_name = Path(self.course_dir).name
        unified = {
            'type': 'root',
            'title': f"{course_name} - Complete Course",
            'children': [],
            'slide_count': total_slides
        }
        for r in successful:
            node = {
                'type': 'category',
                'title': r['hierarchy']['title'],
                'children': r['hierarchy'].get('children', []),
                'slide_count': r['hierarchy'].get('slide_count', r['slides_count']),
                'chapter_info': {
                    'name': r['chapter_name'],
                    'slides_count': r['slides_count']
                }
            }
            unified['children'].append(node)
        return unified

    def save_unified_results(self, unified: Dict[str, Any], results: List[Dict[str, Any]]):
        uni_dir = os.path.join(self.output_dir, "unified")
        # 保存 JSON
        with open(os.path.join(uni_dir, "course_topic_hierarchy.json"), 'w', encoding="utf-8") as f:
            json.dump(unified, f, indent=2, ensure_ascii=False)
        # 可视化
        viz = HierarchyVisualizer()
        viz.save_visualization(unified, uni_dir, "course_")

    def process_course(self):
        pdfs = self.find_pdf_files()
        chapter_results = []
        for pdf in pdfs:
            chapter_results.append(self.process_single_chapter(pdf))

        # 汇总 & 保存
        unified = self.aggregate_chapters(chapter_results)
        self.save_unified_results(unified, chapter_results)
        logger.info(f"[{Path(self.course_dir).name}] Processing completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process multiple courses under a root folder")
    parser.add_argument("--course_dir", required=True,
                        help="Root folder containing multiple course subfolders")
    parser.add_argument("--output_dir", default="course_results",
                        help="Root output folder")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocess all chapters")
    args = parser.parse_args()

    # 如果 course_dir 下有子文件夹，则循环处理每个子文件夹
    entries = sorted(os.listdir(args.course_dir))
    has_subdirs = any(os.path.isdir(os.path.join(args.course_dir, e)) for e in entries)

    if has_subdirs:
        for e in entries:
            sub = os.path.join(args.course_dir, e)
            if os.path.isdir(sub):
                out = os.path.join(args.output_dir, e)
                proc = CourseProcessor(sub, out, args.force)
                proc.process_course()
    else:
        # 直接当作单个课程处理
        proc = CourseProcessor(args.course_dir, args.output_dir, args.force)
        proc.process_course()
