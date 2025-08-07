"""
Module for visualizing the hierarchical topic structure.
Creates tree diagrams.
"""
import os
import json
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchyVisualizer:
    def __init__(self):
        """Initialize the HierarchyVisualizer."""
        pass
    
    def _count_total_slides(self, node: Dict[str, Any]) -> int:
        """Count total number of slides in the hierarchy."""
        # Prefer explicit slide_count, else fallback to counting leaves
        return node.get('slide_count', self._count_leaf_nodes(node))
    
    def _count_leaf_nodes(self, node: Dict[str, Any]) -> int:
        """Count number of leaf nodes (topic groups).

        Gracefully handles missing 'type' or empty children."""
        node_type = node.get('type')
        if node_type == 'leaf':
            return 1
        children = node.get('children') or []
        if children:
            return sum(self._count_leaf_nodes(c) for c in children)
        # No children => treat as leaf
        return 1

    def _get_max_depth(self, node: Dict[str, Any]) -> int:
        """Get maximum depth of the hierarchy.

        Returns 1 for leaf or nodes with no children."""
        node_type = node.get('type')
        children = node.get('children') or []
        if node_type == 'leaf' or not children:
            return 1
        return 1 + max(self._get_max_depth(c) for c in children)

    def generate_text_summary(self, hierarchy: Dict[str, Any]) -> str:
        """Generate a text summary of the hierarchy.
        
        Args:
            hierarchy (Dict): Hierarchy structure
            
        Returns:
            str: Text summary
        """
        # Header
        summary_lines = [
            "TOPIC HIERARCHY SUMMARY",
            "=" * 50,
            f"Total Slides: {self._count_total_slides(hierarchy)}",
            f"Topic Groups: {self._count_leaf_nodes(hierarchy)}",
            f"Hierarchy Levels: {self._get_max_depth(hierarchy)}",
            "",
            "HIERARCHY STRUCTURE:",
            "-" * 30
        ]

        def node_to_text(node: Dict[str, Any], level: int = 0) -> List[str]:
            prefix = "  " * level
            lines: List[str] = []
            node_type = node.get('type')

            if node_type == 'leaf':
                title = node.get('title', 'Untitled')
                count = node.get('slide_count', 1)
                lines.append(f"{prefix}📁 {title} ({count} slides)")
                # clustering-based leaf
                for slide in node.get('slides', []):
                    sid = slide.get('id', 'Unknown')
                    st = slide.get('title', 'Untitled')
                    lines.append(f"{prefix}  • Slide {sid}: {st}")
                # GPT-based leaf
                sid = node.get('slide_id', 'Unknown')
                pn = node.get('page_number', 'Unknown')
                lines.append(f"{prefix}  • Page {pn} (Slide {sid}): {title}")
            else:
                title = node.get('title', 'Untitled')
                count = node.get('slide_count', 0)
                lines.append(f"{prefix}📂 {title} ({count} slides)")
                for child in node.get('children') or []:
                    lines.extend(node_to_text(child, level + 1))
            return lines

        summary_lines.extend(node_to_text(hierarchy))
        return "\n".join(summary_lines)

    def save_visualization(self, hierarchy: Dict[str, Any], output_dir: str, prefix: str = "") -> None:
        """Save text summary to files.
        
        Args:
            hierarchy (Dict): Hierarchy structure
            output_dir (str): Output directory
            prefix (str): Prefix for output filenames
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        text_summary = self.generate_text_summary(hierarchy)
        text_filename = f"{prefix}topic_summary.txt"
        text_path = os.path.join(output_dir, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_summary)
        logger.info(f"Text summary saved to {text_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize topic hierarchy")
    parser.add_argument("--input_file", default="results/topic_hierarchy.json",
                        help="Path to topic hierarchy JSON file")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--output_prefix", default="", help="Prefix for output filenames")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            hierarchy = json.load(f)
    except Exception as e:
        logger.error(f"Error loading hierarchy: {e}")
        return

    viz = HierarchyVisualizer()
    try:
        viz.save_visualization(hierarchy, args.output_dir, args.output_prefix)
        print("\n" + viz.generate_text_summary(hierarchy))
    except Exception as e:
        logger.error(f"Error in visualization: {e}")

if __name__ == "__main__":
    main()
