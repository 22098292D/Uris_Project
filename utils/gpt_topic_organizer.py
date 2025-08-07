# utils/gpt_topic_organizer.py
"""
Module for organizing slide topics using OpenAI GPT to create logical topic hierarchies,
with semantic summarization and title cleaning.
"""
import os
import json
import re
import openai
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTTopicOrganizer:
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 openai_model: str = "gpt-4o-mini"):
        """Initialize the GPT Topic Organizer."""
        self.openai_model = openai_model
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai = True
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key
                self.use_openai = True
            else:
                raise ValueError("OpenAI API key is required for GPT-based topic organization")
    
    def summarize_slides(self, slides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对每张幻灯片内容进行语义总结，生成概念性标题（名词短语），
        覆盖 slide['title']，不含编号或技术标签。
        """
        logger.info(f"Summarizing {len(slides)} slides with GPT...")
        for slide in slides:
            content = slide.get('content', '')[:2000]  # 截断以防过长
            prompt = (
                "Please summarise the theme of the following slide content in a [concise noun phrase],"
                "Do not include numbers such as Lecture, Week, Chapter, etc., and do not use file names or technical tags:\n\n"
                f"{content}\n\n"
                "Output format example:\n\n"
                "Concept: Matrix decomposition method"
            )
            try:
                resp = openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert in extracting themes from academic slides."},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.0
                )
                text = resp.choices[0].message.content.strip()
                # 提取冒号后面的部分
                if 'Concept:' in text:
                    summary = text.split('Concept:')[-1].strip()
                else:
                    summary = text.split(':')[-1].strip()
                # 最终清洗
                slide['title'] = self._clean_title(summary)
            except Exception as e:
                logger.warning(f"Slide {slide.get('id')} summarization failed: {e}")
                # 保留原始标题
        return slides

    def organize_topics(self, slides: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        先语义总结幻灯片标题，再调用 GPT 组织主题层次，
        最后清理所有节点标题并返回树状结构。
        """
        if not self.use_openai:
            raise RuntimeError("OpenAI API is not configured")
        
        # 1. 语义总结每张幻灯片的标题
        slides = self.summarize_slides(slides)
        
        # 2. 准备 GPT 组织提示
        slide_info = [{"id": s['id'], "title": s['title'], "page_number": s.get('page_number', s['id'])}
                      for s in slides]
        prompt = self._create_organization_prompt(slide_info)
        
        # 3. 调用 GPT 获取树状 JSON
        try:
            resp = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            text = resp.choices[0].message.content.strip()
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            tree = json.loads(text[json_start:json_end])
        except Exception as e:
            logger.error(f"GPT 主题组织失败，使用回退方案: {e}")
            tree = self._create_fallback_organization(slides)
        
        # 4. 验证并增强 tree（保留原有逻辑）
        tree = self._validate_and_enhance_tree(tree, slides)
        
        # 5. 递归清理所有节点的标题
        self._clean_tree_titles(tree)
        
        return tree

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert at organising academic presentation slides into a logical hierarchical structure."
            "Please generate a JSON tree based on the slide titles, with node types using "
            "\"category\"(intermediate node) and  \"leaf\"(slide node)。"
            "Leaf nodes must contain slide_id and page_number."
            "Ensure that all node titles are conceptual phrases and do not contain numbers."
        )

    def _create_organization_prompt(self, slide_info: List[Dict]) -> str:
        slides_text = "Slide list:\n"
        for s in slide_info:
            slides_text += f"- Page {s['page_number']}: “{s['title']}” (ID: {s['id']})\n"
        return (
            "Please organise these slides into a hierarchical structure based on logical relationships."
            "Use category and leaf nodes to return pure JSON.\n\n"
            f"{slides_text}\n"
            "Note: Node titles must be conceptual phrases and must not contain numbers or file names.\n"
            "Up to 3 layers.\n\n"
            "Only return JSON, no additional explanation needed."
        )

    def _validate_and_enhance_tree(self, tree: Dict[str, Any], slides: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 原有增强逻辑……
        slide_lookup = {s['id']: s for s in slides}
        def enhance(node):
            if node.get('type') == 'leaf':
                sid = node.get('slide_id')
                if sid in slide_lookup:
                    s = slide_lookup[sid]
                    node.update({
                        'title': s['title'],
                        'content': s.get('content',''),
                        'page_number': s.get('page_number'),
                        'slide_count': 1
                    })
            else:
                count = 0
                for c in node.get('children', []):
                    enhance(c)
                    count += c.get('slide_count',0)
                node['slide_count'] = count
            return node
        return enhance(tree)

    def _create_fallback_organization(self, slides: List[Dict[str, Any]]) -> Dict[str, Any]:
        children = [{
            'type':'leaf',
            'title': s['title'],
            'slide_id': s['id'],
            'page_number': s.get('page_number'),
            'content': s.get('content',''),
            'slide_count':1
        } for s in slides]
        return {'type':'root','title':'Presentation Topics','children':children,'slide_count':len(slides)}

    def _clean_title(self, title: str) -> str:
        """去除编号前缀及多余空白，返回干净的概念性标题。"""
        # 去掉 Lecture/Week/Chapter + 数字
        title = re.sub(r'^(?:Lecture|Week|Chapter)\s*\d+[:\-]?\s*', '', title, flags=re.I)
        # 去除多余标点
        title = re.sub(r'^[\-\–\—\:\s]+', '', title)
        return title.strip()

    def _clean_tree_titles(self, node: Dict[str, Any]) -> None:
        """递归清理树中每个节点的 title。"""
        if 'title' in node:
            node['title'] = self._clean_title(node['title'])
        for c in node.get('children', []):
            self._clean_tree_titles(c)

