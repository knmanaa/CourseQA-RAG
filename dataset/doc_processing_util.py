"""
These util functions are for GENERAL lecture note preprocessings.
For individual courses, plz view the corresponding util.py
"""

import json

from pathlib import Path
from typing import Dict
import pymupdf as pdf

def rename_notes(notes_path: str) -> None:
    notes_path = Path(notes_path)

    items = sorted(notes_path.iterdir(), key=lambda p: p.name.lower())

    # rename each file in a dir to ch1, ch2, ...
    for i, doc_path in enumerate(items, 1):
        print(doc_path)
        new_name = f"{doc_path.parent}/ch{i}.pdf"
        doc_path.rename(new_name)
        print(new_name)


def extract_table(page: pdf.Page) -> list[list[list[str]]]:

    tables = page.find_tables().tables
    
    if len(tables) == 0:
        return []
    
    else:
        result = [table.extract() for table in tables]
        
        # replace \n to whitespace in the cells
        for table in range(len(tables)):
            for row in range(len(result[table])):
                for cell in range(len(result[table][row])):
                    result[table][row][cell] = result[table][row][cell].replace("\n", " ")

        return result
    

def make_one_course_dict(notes_path: str, course_json_target_path = "/shared/project/CourseQA-RAG/dataset/datasets/courses") -> Dict:

    course_dict = {}
    for doc_path in Path(notes_path).iterdir(): # iterate each chapter
        doc = pdf.open(doc_path) # open one chapter

        for page in range(doc.page_count): # iterate each page

            page_name = str(page + 1)
            chapter_name = Path(doc_path).stem
            course_name = Path(notes_path).name

            course_dict_key = course_name + "_" + chapter_name + "_" + page_name
            course_dict_value = doc[page].get_text("text")

            course_dict[course_dict_key] = course_dict_value

    with open(Path(course_json_target_path) / str(course_name + ".json"), "w") as f:
        json.dump(course_dict, f, indent = 4)

    return course_dict

def make_many_course_dict(courses_path = "/shared/project/CourseQA-RAG/data/raw/lectures", course_json_target_path = "/shared/project/CourseQA-RAG/dataset/datasets/courses"):
    courses_path = Path(courses_path)

    result = {}

    for notes_path in courses_path.iterdir():

        if not Path.is_dir(notes_path): continue

        course_dict = make_one_course_dict(notes_path, course_json_target_path)
        result |= course_dict

    with open(Path(course_json_target_path) / "all_courses.json", "w") as f:
        json.dump(result, f, indent = 4)

    return result