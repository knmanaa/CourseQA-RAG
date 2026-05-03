import os
from pathlib import Path

def get_test_files():
    """
    Returns a list of test files for ingestion. 
    If none exist in data/raw/, it automatically creates a dummy text file for testing.
    """
    base_dir = Path("data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Prefer curated JSON corpus first (cleaner than direct PDF extraction).
    json_files = list(base_dir.rglob("*.json"))
    include_all_courses = os.getenv("COURSEQA_INCLUDE_ALL_COURSES_JSON", "0") == "1"
    if not include_all_courses:
        json_files = [p for p in json_files if p.name != "all_courses.json"]

    if json_files:
        return [str(f) for f in sorted(json_files)]

    # Fallback when JSON is unavailable.
    files = list(base_dir.rglob("*.txt")) + list(base_dir.rglob("*.pdf"))
    
    if not files:
        print("Don't find any file in data/raw/. Creating a dummy file to test the pipeline...")
        dummy_file = base_dir / "test_lecture.txt"
        dummy_file.write_text("This is a dummy text file. The capital of France is Paris. RAG (Retrieval-Augmented Generation) is a technique that pipeline a LLM is provided with information from a vector database before answering user's question.", encoding="utf-8")
        files = [dummy_file]
        
    return [str(f) for f in sorted(files)]
