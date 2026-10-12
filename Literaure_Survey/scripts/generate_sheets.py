#!/usr/bin/env python3
"""
generate_sheets.py
Automated generator for Lit Review - WAAM.xlsx across all categories.
Adheres strictly to publication-grade Q1 standards.
"""

import os
import sys
import json
import re
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from db_part1 import PAPERS_P1_P15
from db_part2 import PAPERS_P16_P30
from db_part3 import PAPERS_P31_P45
from db_part4 import PAPERS_P46_P60

def sanitize(val):
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return val
    s = str(val)
    # Remove control characters that corrupt Excel XML
    return ILLEGAL_CHARACTERS_RE.sub("", s)

def create_workbook():
    all_papers = PAPERS_P1_P15 + PAPERS_P16_P30 + PAPERS_P31_P45 + PAPERS_P46_P60
    
    # Also dump updated JSON to workspace root and scripts dir
    for target_dir in [ROOT_DIR, CURRENT_DIR]:
        json_path = os.path.join(target_dir, "papers_db.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_papers, f, indent=4, ensure_ascii=False)
        print(f"Exported JSON database with {len(all_papers)} papers to {json_path}")
    
    wb = openpyxl.Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)
    
    categories = [
        "All Papers",
        "Process",
        "Simulation",
        "Microstructure",
        "Multi-Material",
        "PFM",
        "Ultrasonic"
    ]
    
    headers = [
        "Paper ID",
        "Title",
        "Year",
        "Journal / Source",
        "DOI / Link",
        "Material",
        "Process / Methodology",
        "Key Parameters",
        "Observations",
        "Results (Qualitative & Quantitative)",
        "Possible Research Areas Suggested / Future Work",
        "My Comments"
    ]
    
    # Styles
    header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    
    body_font = Font(name="Arial", size=10, color="1F2937")
    id_font = Font(name="Arial", size=10, bold=True, color="1B365D")
    year_font = Font(name="Arial", size=10, color="4B5563")
    doi_font = Font(name="Arial", size=9, color="0056B3", underline="single")
    
    thin_border = Border(
        left=Side(style="thin", color="D1D5DB"),
        right=Side(style="thin", color="D1D5DB"),
        top=Side(style="thin", color="D1D5DB"),
        bottom=Side(style="thin", color="D1D5DB")
    )
    
    align_left = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_center = Alignment(horizontal="center", vertical="top", wrap_text=True)
    
    alt_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    
    for cat in categories:
        ws = wb.create_sheet(title=cat)
        ws.views.sheetView[0].showGridLines = True
        
        # Append Header
        ws.append(headers)
        ws.row_dimensions[1].height = 32
        
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
            
        # Filter papers for current sheet
        if cat == "All Papers":
            cat_papers = all_papers
        else:
            cat_papers = [p for p in all_papers if p.get("category", "").lower() == cat.lower()]
            
        row_idx = 2
        for p in cat_papers:
            row_data = [
                sanitize(p.get("id", "")),
                sanitize(p.get("title", "")),
                p.get("year", ""),
                sanitize(p.get("journal", "")),
                sanitize(p.get("doi", "")),
                sanitize(p.get("material", "")),
                sanitize(p.get("process_methodology", "")),
                sanitize(p.get("key_params", "")),
                sanitize(p.get("observations", "")),
                sanitize(p.get("results", "")),
                sanitize(p.get("future_work", "")),
                sanitize(p.get("my_comments", ""))
            ]
            ws.append(row_data)
            
            fill_to_use = alt_fill if (row_idx % 2 == 0) else white_fill
            
            # Count max lines in multi-line cells for dynamic row height
            max_lines = 1
            for val in [p.get("observations", ""), p.get("results", ""), p.get("future_work", ""), p.get("my_comments", "")]:
                lines_count = len(str(val).split("\n"))
                if lines_count > max_lines:
                    max_lines = lines_count
            
            # Dynamic row height: approx 16.5pt per bullet line + padding
            ws.row_dimensions[row_idx].height = max(45, max_lines * 17.0)
            
            for col_num in range(1, len(headers) + 1):
                cell = ws.cell(row=row_idx, column=col_num)
                cell.fill = fill_to_use
                cell.border = thin_border
                
                # Column specific alignments and fonts
                if col_num == 1:  # Paper ID
                    cell.font = id_font
                    cell.alignment = align_center
                elif col_num == 3:  # Year
                    cell.font = year_font
                    cell.alignment = align_center
                elif col_num == 5:  # DOI
                    cell.font = doi_font
                    cell.alignment = align_left
                    doi_url = p.get("doi", "")
                    if doi_url and doi_url.startswith("http"):
                        cell.hyperlink = doi_url
                else:
                    cell.font = body_font
                    cell.alignment = align_left
                    
            row_idx += 1
            
        # Freeze panes & autofilter
        ws.freeze_panes = "C2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(row_idx-1, 1)}"
        
        # Column widths
        col_widths = {
            1: 12,   # Paper ID
            2: 36,   # Title
            3: 10,   # Year
            4: 28,   # Journal / Source
            5: 32,   # DOI / Link
            6: 25,   # Material
            7: 38,   # Process / Methodology
            8: 38,   # Key Parameters
            9: 60,   # Observations
            10: 60,  # Results
            11: 45,  # Future Work
            12: 35   # My Comments
        }
        for col_idx, width in col_widths.items():
            ws.column_dimensions[get_column_letter(col_idx)].width = width
            
    excel_path = os.path.join(ROOT_DIR, "Lit Review - WAAM.xlsx")
    wb.save(excel_path)
    print(f"Successfully generated full Excel workbook at {excel_path}")

if __name__ == "__main__":
    create_workbook()
