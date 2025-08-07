# Slides2KGs - Hierarchical Topic Extraction and Evaluation

## Project Structure

```
Slides2KGs/
├── utils/
│   ├── __init__.py           # Utility package initializer
│   ├── slide_preprocessor.py # Extract slides from PDF via MarkItDown
│   ├── gpt_topic_organizer.py# Summarize & organize slides with GPT
│   ├── hierarchy_visualizer.py # Generate text summaries of hierarchies
│   └── text_reshaper.py      # (Optional) Reshape text using T5-based model
├── process_course.py         # Main script: extract, summarize, organize, unify per-course
├── unify_chapters.py         # Merge per-chapter hierarchies into a single JSON
├── json_to_ground_truth.py   # Convert unified JSON to Excel triples (Head, Relation, Tail)
└── kg_similarity.py          # Evaluate KG extraction against ground truth using multiple matching strategies
```

## Dependencies

```bash
pip install openai pandas rapidfuzz sentence-transformers scikit-learn torch transformers
# Also ensure `markitdown` is installed and available on your PATH for PDF→Markdown conversion
```  

## Usage

### 1. Process course slides

Extract slides from PDFs, summarize titles, build hierarchical topics per chapter, and generate unified course hierarchy.

```bash
python process_course.py \
  --course_dir path/to/course_root  
  --output_dir path/to/output_folder [--force]
```

- `--course_dir`: root folder containing one or multiple course subfolders (each with PDFs).  
- `--output_dir`: base output folder where `chapters/` and `unified/` will be created.  
- `--force`: reprocess all chapters even if outputs exist.

After running, you will find in `output_folder`:
```
chapters/             # per-PDF results
  ├── Lecture1/
  │   ├── extracted_slides.json
  │   ├── gpt_topic_hierarchy.json
  │   └── Lecture1_topic_summary.txt
  └── ...
unified/
  ├── course_topic_hierarchy.json
  └── course_topic_summary.txt
```

### 2. (Alternative) Unify existing chapter hierarchies

If you already have per-chapter `gpt_topic_hierarchy.json`, merge them manually:

```bash
python unify_chapters.py \
  --chapters_dir path/to/output_folder/chapters \
  --output_dir path/to/output_folder/unified
```

### 3. Generate ground-truth Excel from JSON

Convert each course's `course_topic_hierarchy.json` into an Excel file of triples:

```bash
python json_to_ground_truth.py
```

- The script reads `ROOT_DIR = "course_resultsV2"` by default.  
- It outputs `extractedV2.xlsx` under each course folder.

### 4. Evaluate extracted KG against ground truth

Compute precision, recall, and F1 using fuzzy, semantic, and GPT-based matching:

```bash
python kg_similarity.py
```

- Edit the script to adjust `df_gt` and `df_pred` file paths if needed.  
- Ensure `OPENAI_API_KEY` is set in your environment for GPT checks.

## Notes

- To improve slide text quality, enable the `TextReshaper` in `process_course.py` by initializing `SlideExtractor(use_reshaper=True)`.  
- All utility modules under `utils/` are imported by the main scripts.
- Customize GPT model names and thresholds in `gpt_topic_organizer.py` and `kg_similarity.py` as required.

