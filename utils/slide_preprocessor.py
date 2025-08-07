"""
Module for extracting text from PDF slides using MarkItDown.
Modified to extract title and content for each slide for clustering-based organization.
"""
import os
import sys
import subprocess
import tempfile
import json
from typing import List, Optional, Dict, Tuple
import logging
import re

# Handle imports for both standalone and package usage
try:
    from .text_reshaper import TextReshaper
except ImportError:
    # For standalone execution, add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from slides2kgs.utils.text_reshaper import TextReshaper
    except ImportError:
        # If still fails, create a dummy class
        class TextReshaper:
            def __init__(self, *args, **kwargs):
                pass
            def reshape_batch(self, texts):
                return texts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlideExtractor:
    def __init__(self, use_reshaper: bool = False, model_name: str = "t5-base"):
        """Initialize the SlideExtractor with MarkItDown.
        
        Args:
            use_reshaper (bool): Whether to use the TextReshaper for improving text quality
            model_name (str): Name of the pretrained model to use for reshaping
        """
        self.use_reshaper = use_reshaper
        if use_reshaper:
            try:
                self.reshaper = TextReshaper(model_name=model_name)
                logger.info("TextReshaper initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TextReshaper: {str(e)}")
                logger.warning("Continuing without text reshaping")
                self.use_reshaper = False
    
    def _run_markitdown(self, pdf_path: str, output_path: str) -> bool:
        """Run MarkItDown to convert PDF to markdown.
        
        Args:
            pdf_path (str): Path to the input PDF file
            output_path (str): Path to the output markdown file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Run markitdown command
            cmd = ["markitdown", pdf_path, "-o", output_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"MarkItDown conversion successful")
            logger.debug(f"MarkItDown output: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"MarkItDown conversion failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("MarkItDown command not found. Please ensure it's installed and in PATH.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running MarkItDown: {str(e)}")
            return False
    
    def _extract_title_from_slide(self, slide_content: str) -> str:
        """Extract title from slide content.
        
        The first sentence/line is typically the title in PPT slides.
        
        Args:
            slide_content (str): Raw slide content
            
        Returns:
            str: Extracted title (first meaningful sentence)
        """
        lines = slide_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip common noise patterns
            if (line.isdigit() or  # Page numbers
                'New Jersey Institute of Technology' in line or  # Footer
                'The Hong Kong Polytechnic University' in line or
                re.match(r'^[=\-\*]+$', line) or  # Separators
                line.lower() in ['comp4434']):  # Course code alone
                continue
            
            # Strip markdown headers if present
            if line.startswith('#'):
                line = line.lstrip('#').strip()
            
            # This is the first meaningful line - use it as title
            if len(line) > 0:
                # Clean up the title
                title = line
                
                # If title is too long, try to find the first sentence
                if len(title) > 100:
                    sentences = re.split(r'[.!?]+', title)
                    if sentences and len(sentences[0].strip()) > 0:
                        title = sentences[0].strip()
                
                # Remove common prefixes that might not be actual titles
                title = re.sub(r'^(slide\s*\d+:?\s*)', '', title, flags=re.IGNORECASE)
                
                return title[:100]  # Limit to 100 characters
        
        return "Untitled Slide"
    
    def _clean_content(self, content: str, title: str) -> str:
        """Clean slide content by removing title and noise.
        
        Args:
            content (str): Raw slide content
            title (str): Extracted title
            
        Returns:
            str: Cleaned content
        """
        lines = content.strip().split('\n')
        cleaned_lines = []
        title_found = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip the title line (once we find it)
            if not title_found and (line == title or title in line):
                title_found = True
                continue
                
            # Skip university footer
            if 'The Hong Kong Polytechnic University' in line:
                continue
                
            # Skip page numbers
            if line.isdigit():
                continue
                
            # Skip slide separators
            if re.match(r'^[=\-\*]+$', line):
                continue
                
            # Skip markdown headers if they match title
            if line.startswith('#') and title in line:
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _parse_markdown_to_slides(self, markdown_content: str) -> List[Dict[str, any]]:
        """Parse markdown content to extract slides with titles and content.
        
        Args:
            markdown_content (str): Markdown content from MarkItDown
            
        Returns:
            List[Dict]: List of slide dictionaries with id, title, and content
        """
        slides = []
        lines = markdown_content.split('\n')
        
        current_slide_id = 1
        current_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines at the beginning of a slide
            if not line and not current_content:
                i += 1
                continue
            
            # Check for university footer (indicates end of slide)
            if 'New Jersey Institute of Technology' in line:
                # Look ahead for page number on next line
                page_num = None
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.isdigit() and len(next_line) <= 2:  # Real page numbers are 1-2 digits
                        page_num = int(next_line)
                        i += 1  # Skip the page number line
                
                # Process current slide if we have content
                if current_content:
                    slide_text = '\n'.join(current_content).strip()
                    if slide_text:
                        title = self._extract_title_from_slide(slide_text)
                        content = self._clean_content(slide_text, title)
                        # if title == "Untitled Slide":
                        #     continue
                        slides.append({
                            'id': current_slide_id,
                            'title': title,
                            'content': content,
                            'page_number': page_num if page_num else current_slide_id
                        })
                    current_slide_id += 1
                    current_content = []
                i += 1
                continue
            
            # Add content to current slide
            current_content.append(lines[i])
            i += 1
        
        # Add the last slide if there's content (this handles cases where there's no footer at the end)
        if current_content:
            slide_text = '\n'.join(current_content).strip()
            # Only add if there's meaningful content (not just university name)
            if slide_text and not slide_text.strip() == 'New Jersey Institute of Technology':
                title = self._extract_title_from_slide(slide_text)
                content = self._clean_content(slide_text, title)
                slides.append({
                    'id': current_slide_id,
                    'title': title,
                    'content': content,
                    'page_number': current_slide_id
                })
        
        # If we still only have one slide but content is very long, try to split by headers
        if len(slides) == 1 and len(markdown_content) > 5000:
            single_slide = slides[0]
            header_slides = self._split_by_headers(single_slide['content'])
            if len(header_slides) > 1:
                slides = []
                for i, (title, content) in enumerate(header_slides, 1):
                    slides.append({
                        'id': i,
                        'title': title,
                        'content': content,
                        'page_number': i
                    })
        
        return slides
    
    def _split_by_headers(self, content: str) -> List[Tuple[str, str]]:
        """Split content by headers to create slides."""
        slides = []
        lines = content.split('\n')
        current_slide_content = []
        current_title = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for major headers
            if (line_stripped.startswith('#') or 
                (len(line_stripped) > 0 and 
                 any(keyword in line_stripped.lower() for keyword in 
                     ['knowledge graph', 'limitation', 'what do we have', 'publications', 
                      'question answering', 'education', 'recommender', 'introduction',
                      'methodology', 'results', 'conclusion', 'future work']))):
                
                # Save previous slide
                if current_title and current_slide_content:
                    content_text = '\n'.join(current_slide_content).strip()
                    if content_text:
                        slides.append((current_title, content_text))
                
                # Start new slide
                current_title = line_stripped.lstrip('#').strip()
                current_slide_content = []
            else:
                if current_title:  # Only add content if we have a title
                    current_slide_content.append(line)
        
        # Add the last slide
        if current_title and current_slide_content:
            content_text = '\n'.join(current_slide_content).strip()
            if content_text:
                slides.append((current_title, content_text))
        
        return slides if len(slides) > 1 else [("Main Content", content)]

    def extract_slides(self, pdf_path: str) -> List[Dict[str, any]]:
        """Extract slides with titles and content from PDF file using MarkItDown.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Dict]: List of slide dictionaries with id, title, content, and page_number
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create a temporary file for markdown output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_md_path = temp_file.name
        
        try:
            # Convert PDF to markdown using MarkItDown
            logger.info(f"Converting PDF to markdown using MarkItDown: {pdf_path}")
            
            if not self._run_markitdown(pdf_path, temp_md_path):
                raise RuntimeError("Failed to convert PDF using MarkItDown")
            
            # Read the markdown content
            with open(temp_md_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            logger.info(f"Successfully converted PDF to markdown ({len(markdown_content)} characters)")
            
            # Parse markdown content into slides
            slides_data = self._parse_markdown_to_slides(markdown_content)
            logger.info(f"Parsed {len(slides_data)} slides from markdown content")
            
            # Apply text reshaping if enabled
            if self.use_reshaper and slides_data:
                logger.info("Reshaping slide titles and content...")
                
                # Reshape titles
                titles = [slide['title'] for slide in slides_data]
                reshaped_titles = self.reshaper.reshape_batch(titles)
                
                # Reshape content
                contents = [slide['content'] for slide in slides_data]
                reshaped_contents = self.reshaper.reshape_batch(contents)
                
                # Update slides with reshaped text
                for i, slide in enumerate(slides_data):
                    slide['title'] = reshaped_titles[i]
                    slide['content'] = reshaped_contents[i]
            
            return slides_data
            
        except Exception as e:
            logger.error(f"Error processing PDF with MarkItDown: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_md_path)
            except:
                pass

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract slides with titles and content from PDF")
    parser.add_argument("--input_file", required=True, help="Path to input PDF file")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--use_reshaper", action="store_true", help="Use text reshaper")
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize extractor
    extractor = SlideExtractor(use_reshaper=args.use_reshaper)
    
    try:
        # Extract slides
        slides = extractor.extract_slides(args.input_file)
        
        # Save the extracted slides to a JSON file
        output_path = os.path.join(args.output_dir, "extracted_slides.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(slides, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully extracted {len(slides)} slides")
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        for slide in slides:
            logger.info(f"Slide {slide['id']}: {slide['title'][:50]}...")
        
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")

if __name__ == "__main__":
    main() 