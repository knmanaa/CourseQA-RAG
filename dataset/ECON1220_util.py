"""
THIS UTIL IS TEMPERARY NOT IN USE.

These util functions are specially for ECON1220 lecture notes
Using these func on other materials could be not working.
"""

import doc_processing_util
import re
import json

from pathlib import Path
from typing import Dict
import pymupdf as pdf

def extract_text(page: pdf.Page) -> list[str]:

    # lecture notes always use this "\uf06e" to separate ideas
    mid_process = page.get_text("text").split("\uf06e")

    # Remove the unnecessary title in lecture note if have this
    try:
        pattern = r"\nECON1220 L1 – ([\w ]+?)\n(\d+)\n"
        unneed_title_start, unneed_title_end = re.search(pattern, mid_process[-1]).span()
        mid_process[-1] = mid_process[-1][:unneed_title_start] + mid_process[-1][unneed_title_end:]
    except:
        ...

    # lecture notes always use this ""\uf06c"" to separate ideas
    temp = []
    for phrase in mid_process:
        temp.extend(phrase.split("\uf06c"))

    # filter away \n
    result = [phrase.replace("\n", "") for phrase in temp]
    return result

def make_page_dict(page: pdf.Page) -> Dict:

    # return value
    return {"text": extract_text(page),  "table" : doc_processing_util.extract_table(page)}

def make_course_dict(notes_path: str, course_json_target_path = "/shared/project/CourseQA-RAG/dataset") -> Dict:
    
    # iterate each chapter
    course_value = {}
    for doc_path in Path(notes_path).iterdir():
        doc = pdf.open(doc_path) # open one chapter

        chapter_value = {str(page + 1) : make_page_dict(doc[page]) for page in range(doc.page_count)}

        chapter_key = Path(doc_path).stem

        course_value[chapter_key] = chapter_value

    course_key = Path(notes_path).name
    course = {course_key : course_value}

    with open(Path(course_json_target_path) / str(course_key + ".json"), "w") as f:
        json.dump(course, f, indent = 4)

    return course


