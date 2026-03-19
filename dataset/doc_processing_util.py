"""
These util functions are for GENERAL lecture note preprocessings.
For individual courses, plz view the corresponding util.py
"""

import json
import pymupdf as pdf

def extract_text_outside_tables():
    print("balabababa")

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

testing = {
    "Course 1 Name" : {
        1 : {
            "text" : ["Apple", "Banana", "Cucumber"],
            "table": [["English", "German", "Vietnamese"], ["Hello", "Hallo", "Xin chao"]]
        }, 

        2 : {
            "text" : ["Dog", "Egg", "Flower"],
            "table": [["English", "German", "Vietnamese"], ["1", "2", "3"]]
        }
    }, 

    "Course 2 Name" : {
        1 : {
            "text" : ["Apple", "Banana", "Cucumber"],
            "table": [["English", "German", "Vietnamese"], ["Bye", "bye bye", "BYE BYE BYE"]]
        }, 

        2 : {
            "text" : ["Dog", "Egg", "Flower"],
            "table": [["English", "German", "Vietnamese"], ["10", "20", "30"]]
        }
    }
}

with open("/shared/project/CourseQA-RAG/dataset/testing.json", "w", encoding = "utf-8") as f:
    json.dump(testing, f, indent = 4)



