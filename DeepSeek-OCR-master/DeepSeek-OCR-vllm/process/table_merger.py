"""
Table Merger for Cross-Page Tables
This module detects and merges tables that are split across multiple pages in PDF OCR output.
"""
import re
from typing import List, Tuple, Optional


class TableBlock:
    """Represents a table block in the document."""
    
    def __init__(self, content: str, page_num: int, start_pos: int, end_pos: int):
        self.content = content
        self.page_num = page_num
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.rows = self._parse_rows()
        self.is_complete = self._check_completeness()
        
    def _parse_rows(self) -> List[str]:
        """Parse table rows from markdown content."""
        lines = self.content.strip().split('\n')
        rows = [line.strip() for line in lines if line.strip().startswith('|')]
        return rows
    
    def _check_completeness(self) -> bool:
        """Check if table appears complete."""
        if len(self.rows) < 2:
            return False
        
        has_separator = any(
            re.match(r'^\|[\s\-:]+\|', row) and set(row.replace('|', '').replace(' ', '').replace(':', '')) <= {'-'}
            for row in self.rows[:3]
        )
        
        return has_separator
    
    def get_column_count(self) -> int:
        """Get the number of columns in the table."""
        if not self.rows:
            return 0
        return max(row.count('|') - 1 for row in self.rows)
    
    def get_header(self) -> Optional[str]:
        """Get the header row if exists."""
        if len(self.rows) >= 1:
            return self.rows[0]
        return None


def extract_tables(content: str) -> List[Tuple[int, int, str]]:
    """Extract all markdown tables from content."""
    tables = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('|') and '|' in line[1:]:
            table_start = i
            table_lines = [lines[i]]
            i += 1
            
            while i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith('|') and '|' in next_line[1:]:
                    table_lines.append(lines[i])
                    i += 1
                elif next_line == '':
                    lookahead = i + 1
                    while lookahead < len(lines) and lines[lookahead].strip() == '':
                        lookahead += 1
                    
                    if lookahead < len(lines) and lines[lookahead].strip().startswith('|'):
                        table_lines.append(lines[i])
                        i += 1
                    else:
                        break
                else:
                    break
            
            table_content = '\n'.join(table_lines)
            char_start = sum(len(lines[j]) + 1 for j in range(table_start))
            char_end = char_start + len(table_content)
            
            tables.append((char_start, char_end, table_content))
        else:
            i += 1
    
    return tables


def can_merge_tables(table1: TableBlock, table2: TableBlock) -> bool:
    """Determine if two tables from consecutive pages can be merged."""
    if table2.page_num != table1.page_num + 1:
        return False
    
    if table1.get_column_count() != table2.get_column_count():
        return False
    
    if table1.is_complete and table2.is_complete:
        header1 = table1.get_header()
        header2 = table2.get_header()
        
        if header1 and header2:
            similarity = _calculate_similarity(header1, header2)
            if similarity > 0.8:
                return True
    
    if not table1.is_complete:
        return True
    
    if not table2.is_complete and len(table2.rows) > 0:
        return True
    
    return False


def merge_tables(table1: TableBlock, table2: TableBlock) -> str:
    """Merge two tables into one markdown table."""
    rows1 = table1.rows.copy()
    rows2 = table2.rows.copy()
    
    separator_idx1 = -1
    for i, row in enumerate(rows1):
        if re.match(r'^\|[\s\-:]+\|', row) and set(row.replace('|', '').replace(' ', '').replace(':', '')) <= {'-'}:
            separator_idx1 = i
            break
    
    separator_idx2 = -1
    for i, row in enumerate(rows2):
        if re.match(r'^\|[\s\-:]+\|', row) and set(row.replace('|', '').replace(' ', '').replace(':', '')) <= {'-'}:
            separator_idx2 = i
            break
    
    merged_rows = []
    
    if separator_idx1 >= 0:
        merged_rows.extend(rows1[:separator_idx1 + 1])
        merged_rows.extend(rows1[separator_idx1 + 1:])
    else:
        merged_rows.extend(rows1)
    
    if separator_idx2 >= 0:
        merged_rows.extend(rows2[separator_idx2 + 1:])
    else:
        if len(rows2) > 0 and len(rows1) > 0:
            similarity = _calculate_similarity(rows1[0], rows2[0])
            if similarity > 0.8:
                merged_rows.extend(rows2[1:])
            else:
                merged_rows.extend(rows2)
        else:
            merged_rows.extend(rows2)
    
    return '\n'.join(merged_rows)


def _calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings."""
    if not str1 or not str2:
        return 0.0
    
    s1 = ' '.join(str1.lower().split())
    s2 = ' '.join(str2.lower().split())
    
    if s1 == s2:
        return 1.0
    
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def merge_cross_page_tables(pages_content: List[str]) -> List[str]:
    """Main function to merge cross-page tables in PDF OCR output."""
    all_tables = []
    for page_num, page_content in enumerate(pages_content):
        tables = extract_tables(page_content)
        for start_pos, end_pos, table_content in tables:
            table_block = TableBlock(table_content, page_num, start_pos, end_pos)
            all_tables.append(table_block)
    
    merge_groups = []
    merged_indices = set()
    
    for i in range(len(all_tables) - 1):
        if i in merged_indices:
            continue
            
        current_group = [i]
        j = i + 1
        
        while j < len(all_tables):
            if can_merge_tables(all_tables[current_group[-1]], all_tables[j]):
                current_group.append(j)
                merged_indices.add(j)
                j += 1
            else:
                break
        
        if len(current_group) > 1:
            merge_groups.append(current_group)
            merged_indices.add(i)
    
    result_pages = []
    
    for page_num, page_content in enumerate(pages_content):
        page_tables = [t for t in all_tables if t.page_num == page_num]
        
        if not page_tables:
            result_pages.append(page_content)
            continue
        
        modified_content = page_content
        
        for table in sorted(page_tables, key=lambda t: t.start_pos, reverse=True):
            table_idx = all_tables.index(table)
            
            merge_group = None
            for group in merge_groups:
                if table_idx in group:
                    merge_group = group
                    break
            
            if merge_group and table_idx == merge_group[0]:
                tables_to_merge = [all_tables[idx] for idx in merge_group]
                merged_table = tables_to_merge[0].content
                
                for next_table in tables_to_merge[1:]:
                    merged_table = merge_tables(
                        TableBlock(merged_table, 0, 0, 0),
                        next_table
                    )
                
                modified_content = (
                    modified_content[:table.start_pos] +
                    merged_table +
                    modified_content[table.end_pos:]
                )
            elif merge_group and table_idx in merge_group[1:]:
                modified_content = (
                    modified_content[:table.start_pos] +
                    modified_content[table.end_pos:]
                )
        
        result_pages.append(modified_content)
    
    return result_pages


def process_pdf_output(content: str, page_separator: str = '<--- Page Split --->') -> str:
    """Process complete PDF OCR output and merge cross-page tables."""
    pages = content.split(f'\n{page_separator}\n')
    merged_pages = merge_cross_page_tables(pages)
    return f'\n{page_separator}\n'.join(merged_pages)


def process_pdf_output_with_stats(content: str, page_separator: str = '<--- Page Split --->') -> Tuple[str, dict]:
    """
    Process complete PDF OCR output and merge cross-page tables with statistics.
    
    Returns:
        Tuple of (merged_content, statistics_dict)
        
    Statistics dict contains:
        - total_tables: Total number of tables found
        - merged_count: Number of table groups that were merged
        - details: List of merge details with page numbers and previews
    """
    pages = content.split(f'\n{page_separator}\n')
    
    # Parse all tables
    all_tables = []
    for page_num, page_content in enumerate(pages):
        tables = extract_tables(page_content)
        for start_pos, end_pos, table_content in tables:
            table_block = TableBlock(table_content, page_num, start_pos, end_pos)
            all_tables.append(table_block)
    
    # Identify tables to merge
    merge_groups = []
    merged_indices = set()
    
    for i in range(len(all_tables) - 1):
        if i in merged_indices:
            continue
            
        current_group = [i]
        j = i + 1
        
        while j < len(all_tables):
            if can_merge_tables(all_tables[current_group[-1]], all_tables[j]):
                current_group.append(j)
                merged_indices.add(j)
                j += 1
            else:
                break
        
        if len(current_group) > 1:
            merge_groups.append(current_group)
            merged_indices.add(i)
    
    # Merge tables
    merged_pages = merge_cross_page_tables(pages)
    merged_content = f'\n{page_separator}\n'.join(merged_pages)
    
    # Build statistics
    stats = {
        'total_tables': len(all_tables),
        'merged_count': len(merge_groups),
        'details': []
    }
    
    for group in merge_groups:
        first_table = all_tables[group[0]]
        last_table = all_tables[group[-1]]
        header = first_table.get_header() or 'Unknown table'
        # Truncate header for preview
        header_preview = header[:50] + '...' if len(header) > 50 else header
        
        stats['details'].append({
            'from_page': first_table.page_num + 1,  # 1-indexed for user display
            'to_page': last_table.page_num + 1,
            'table_count': len(group),
            'table_preview': header_preview
        })
    
    return merged_content, stats


def extract_html_tables(content: str) -> List[Tuple[int, int, str]]:
    """
    Extract HTML format tables from content.
    Returns list of (start_pos, end_pos, table_content).
    """
    import re
    tables = []
    
    # Pattern to match <table>...</table> including attributes
    pattern = r'<table[^>]*>.*?</table>'
    
    for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
        start_pos = match.start()
        end_pos = match.end()
        table_content = match.group(0)
        tables.append((start_pos, end_pos, table_content))
    
    return tables


class HTMLTableBlock:
    """Represents an HTML table block in the document."""
    
    def __init__(self, content: str, page_num: int, start_pos: int, end_pos: int):
        self.content = content
        self.page_num = page_num
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.rows = self._parse_rows()
        
    def _parse_rows(self) -> List[str]:
        """Parse table rows from HTML content."""
        import re
        # Extract all <tr>...</tr> blocks
        tr_pattern = r'<tr[^>]*>.*?</tr>'
        rows = re.findall(tr_pattern, self.content, re.DOTALL | re.IGNORECASE)
        return rows
    
    def get_column_count(self) -> int:
        """Get the number of columns by counting <td> or <th> in first row."""
        if not self.rows:
            return 0
        import re
        # Count <td> and <th> tags
        td_count = len(re.findall(r'<td[^>]*>', self.rows[0], re.IGNORECASE))
        th_count = len(re.findall(r'<th[^>]*>', self.rows[0], re.IGNORECASE))
        return max(td_count, th_count)


def can_merge_html_tables(table1: HTMLTableBlock, table2: HTMLTableBlock) -> bool:
    """Determine if two HTML tables from consecutive pages can be merged."""
    if table2.page_num != table1.page_num + 1:
        return False
    
    if table1.get_column_count() != table2.get_column_count():
        return False
    
    # HTML tables are easier - just check column count and page adjacency
    return True


def merge_html_tables(table1: HTMLTableBlock, table2: HTMLTableBlock) -> str:
    """Merge two HTML tables."""
    import re
    
    # Extract opening tag from table1
    table1_opening = re.match(r'<table[^>]*>', table1.content, re.IGNORECASE).group(0)
    
    # Extract all rows
    rows1 = table1.rows
    rows2 = table2.rows
    
    # Combine rows
    all_rows = rows1 + rows2
    
    # Reconstruct table
    merged_table = table1_opening + '\n' + '\n'.join(all_rows) + '\n</table>'
    
    return merged_table

