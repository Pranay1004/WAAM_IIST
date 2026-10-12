#!/usr/bin/env python3
"""
build_full_database.py
Combines db_part1, db_part2, db_part3, and db_part4 into papers_db.json.
"""
import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from db_part1 import PAPERS_P1_P15
from db_part2 import PAPERS_P16_P30
from db_part3 import PAPERS_P31_P45
from db_part4 import PAPERS_P46_P60

def build_database():
    all_papers = PAPERS_P1_P15 + PAPERS_P16_P30 + PAPERS_P31_P45 + PAPERS_P46_P60
    output_path = os.path.join(ROOT_DIR, "papers_db.json")
    scripts_json_path = os.path.join(CURRENT_DIR, "papers_db.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=4, ensure_ascii=False)
    with open(scripts_json_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=4, ensure_ascii=False)
    print(f"Successfully compiled {len(all_papers)} papers into {output_path} and {scripts_json_path}")
    return all_papers

if __name__ == "__main__":
    build_database()
