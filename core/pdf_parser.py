"""
PDF Dataset Parser with proper file handling
"""

import pdfplumber
import re
from typing import Dict, List, Any
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse traffic dataset PDFs"""
    
    def __init__(self):
        self.data = {
            'metadata': {},
            'violations': [],
            'traffic_volume': [],
            'intersection_info': {},
            'pedestrian_data': []
        }
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF file and extract structured data"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Parsing PDF: {file_path}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                self.data['metadata'] = {
                    'pages': len(pdf.pages),
                    'file_name': os.path.basename(file_path),
                    'parsed_at': datetime.now().isoformat()
                }
                
                full_text = ""
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    full_text += text
                    
                    tables = page.extract_tables()
                    for table in tables:
                        self._process_table(table)
                
                self._extract_text_info(full_text)
                
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise
        
        return self.data
    
    def _process_table(self, table: List[List]) -> None:
        """Process a table from PDF"""
        if not table or len(table) < 2:
            return
        
        headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
        
        for row in table[1:]:
            if not any(row):
                continue
            
            row_dict = {}
            for i, cell in enumerate(row):
                if i < len(headers) and cell:
                    row_dict[headers[i]] = str(cell).strip()
            
            if row_dict:
                self._categorize_row(row_dict)
    
    def _categorize_row(self, row: Dict) -> None:
        """Categorize row into appropriate data type"""
        row_text = str(row).lower()
        
        if any(k in row_text for k in ['violation', 'block', 'vehicle_id', 'id']):
            self.data['violations'].append(row)
        elif any(k in row_text for k in ['volume', 'count', 'hourly', 'traffic']):
            self.data['traffic_volume'].append(row)
        elif any(k in row_text for k in ['pedestrian', 'crossing']):
            self.data['pedestrian_data'].append(row)
    
    def _extract_text_info(self, text: str) -> None:
        """Extract information from text"""
        text_lower = text.lower()
        
        # Extract intersection name
        patterns = [
            r'intersection[:\s]+([A-Za-z\s&]+)',
            r'location[:\s]+([A-Za-z\s&]+)',
            r'banashankari',
            r'junction'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.data['intersection_info']['name'] = match.group(0).title()
                break
        
        if 'intersection_info' not in self.data or not self.data['intersection_info']:
            self.data['intersection_info']['name'] = "Banashankari Junction"
        
        # Extract violation counts
        count_patterns = [
            r'(\d+)\s*violations?',
            r'(\d+)\s*vehicles?\s*blocking',
            r'total.*?(\d+)\s*incidents'
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    self.data['metadata']['violation_count'] = int(match.group(1))
                    break
                except ValueError:
                    pass