import re
from typing import List, Tuple, Optional, Dict


class TableBlock:
    def __init__(self, content: str, page_num: int, start_pos: int, end_pos: int):
        self.content = content
        self.page_num = page_num
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.rows = self._parse_rows()
        self.is_complete = self._check_completeness()
        
    def _parse_rows(self) -> List[str]:
        lines = self.content.strip().split('\n')
        rows = [line.strip() for line in lines if line.strip().startswith('|')]
        return rows
    
    def _check_completeness(self) -> bool:
        if len(self.rows) < 2:
            return False
        
        has_separator = any(
            re.match(r'^\|[\s\-:]+\|', row) and 
            set(row.replace('|', '').replace(' ', '').replace(':', '')) <= {'-'}
            for row in self.rows[:3]
        )
        
        return has_separator
    
    def get_column_count(self) -> int:
        if not self.rows:
            return 0
        return max(row.count('|') - 1 for row in self.rows)
    
    def get_header(self) -> Optional[str]:
        if len(self.rows) >= 1:
            return self.rows[0]
        return None


def extract_tables(content: str) -> List[Tuple[int, int, str]]:
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
            char_start = sum(len(line) + 1 for line in lines[:table_start])
            char_end = char_start + len(table_content)
            
            tables.append((char_start, char_end, table_content))
        else:
            i += 1
    
    return tables


def _calculate_similarity(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    
    str1_clean = re.sub(r'[^a-zA-Z0-9]', '', str1.lower())
    str2_clean = re.sub(r'[^a-zA-Z0-9]', '', str2.lower())
    
    if not str1_clean or not str2_clean:
        return 0.0
    
    matches = sum(1 for a, b in zip(str1_clean, str2_clean) if a == b)
    max_len = max(len(str1_clean), len(str2_clean))
    
    return matches / max_len if max_len > 0 else 0.0


def can_merge_tables(table1: TableBlock, table2: TableBlock) -> bool:
    if table2.page_num != table1.page_num + 1:
        return False
    
    if table1.get_column_count() != table2.get_column_count():
        return False
    
    if not table1.is_complete:
        return True
    
    header1 = table1.get_header()
    header2 = table2.get_header()
    
    if header1 and header2:
        similarity = _calculate_similarity(header1, header2)
        return similarity >= 0.8
    
    return False


def merge_tables(table1: TableBlock, table2: TableBlock) -> str:
    rows1 = table1.rows.copy()
    rows2 = table2.rows.copy()
    
    has_separator2 = any(
        set(row.replace('|', '').replace(' ', '').replace(':', '')) <= {'-'}
        for row in rows2[:3]
    )
    
    if has_separator2:
        rows2 = [row for row in rows2 if not (
            set(row.replace('|', '').replace(' ', '').replace(':', '')) <= {'-'}
        )]
    
    header1 = rows1[0] if rows1 else None
    header2 = rows2[0] if rows2 else None
    
    if header1 and header2:
        similarity = _calculate_similarity(header1, header2)
        if similarity >= 0.8:
            rows2 = rows2[1:]
    
    merged_rows = rows1 + rows2
    return '\n'.join(merged_rows)


def _parse_all_tables(pages_content: List[str]) -> List[TableBlock]:
    all_tables = []
    
    for page_num, page_content in enumerate(pages_content):
        tables_in_page = extract_tables(page_content)
        
        for start_pos, end_pos, table_content in tables_in_page:
            table_block = TableBlock(table_content, page_num, start_pos, end_pos)
            all_tables.append(table_block)
    
    return all_tables


def _identify_merge_groups(all_tables: List[TableBlock]) -> List[List[int]]:
    merge_groups = []
    i = 0
    
    while i < len(all_tables):
        current_group = [i]
        
        j = i + 1
        while j < len(all_tables):
            if can_merge_tables(all_tables[current_group[-1]], all_tables[j]):
                current_group.append(j)
                j += 1
            else:
                break
        
        if len(current_group) > 1:
            merge_groups.append(current_group)
        
        i = current_group[-1] + 1
    
    return merge_groups


def merge_cross_page_tables(pages_content: List[str]) -> List[str]:
    all_tables = _parse_all_tables(pages_content)
    merge_groups = _identify_merge_groups(all_tables)
    
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
    pages = content.split(f'\n{page_separator}\n')
    merged_pages = merge_cross_page_tables(pages)
    
    try:
        from config import ENABLE_HTML_TABLE_MERGE
        if ENABLE_HTML_TABLE_MERGE:
            merged_pages = merge_cross_page_html_tables(merged_pages)
    except:
        pass
    
    return f'\n{page_separator}\n'.join(merged_pages)


def process_pdf_output_with_stats(content: str, page_separator: str = '<--- Page Split --->') -> Tuple[str, Dict]:
    pages = content.split(f'\n{page_separator}\n')
    
    all_tables = _parse_all_tables(pages)
    merge_groups = _identify_merge_groups(all_tables)
    
    merged_pages = merge_cross_page_tables(pages)
    
    stats = {
        'total_markdown_tables': len(all_tables),
        'markdown_merged_count': len(merge_groups),
        'total_html_tables': 0,
        'html_merged_count': 0,
        'merged_count': len(merge_groups),
        'total_tables': len(all_tables),
        'details': []
    }
    
    for group in merge_groups:
        first_table = all_tables[group[0]]
        last_table = all_tables[group[-1]]
        header = first_table.get_header() or 'Unknown table'
        header_preview = header[:50] + '...' if len(header) > 50 else header
        
        stats['details'].append({
            'type': 'markdown',
            'from_page': first_table.page_num + 1,
            'to_page': last_table.page_num + 1,
            'table_count': len(group),
            'table_preview': header_preview
        })
    
    try:
        from config import ENABLE_HTML_TABLE_MERGE
        if ENABLE_HTML_TABLE_MERGE:
            html_tables = _parse_html_tables(merged_pages)
            html_merge_groups = _identify_html_merge_groups(html_tables)
            merged_pages = merge_cross_page_html_tables(merged_pages)
            
            stats['total_html_tables'] = len(html_tables)
            stats['html_merged_count'] = len(html_merge_groups)
            stats['merged_count'] += len(html_merge_groups)
            stats['total_tables'] = stats['total_markdown_tables'] + len(html_tables)
            
            for group in html_merge_groups:
                first_table = html_tables[group[0]]
                last_table = html_tables[group[-1]]
                preview = first_table.rows[0] if first_table.rows else 'HTML table'
                preview = preview[:50] + '...' if len(preview) > 50 else preview
                
                stats['details'].append({
                    'type': 'html',
                    'from_page': first_table.page_num + 1,
                    'to_page': last_table.page_num + 1,
                    'table_count': len(group),
                    'table_preview': preview
                })
    except:
        pass
    
    merged_content = f'\n{page_separator}\n'.join(merged_pages)
    return merged_content, stats


# HTML table processing
def extract_html_tables(content: str) -> List[Tuple[int, int, str]]:
    import re
    tables = []
    pattern = r'<table[^>]*>.*?</table>'
    
    for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
        start_pos = match.start()
        end_pos = match.end()
        table_content = match.group(0)
        tables.append((start_pos, end_pos, table_content))
    
    return tables


class HTMLTableBlock:
    def __init__(self, content: str, page_num: int, start_pos: int, end_pos: int):
        self.content = content
        self.page_num = page_num
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.rows = self._parse_rows()
        
    def _parse_rows(self) -> List[str]:
        import re
        tr_pattern = r'<tr[^>]*>.*?</tr>'
        rows = re.findall(tr_pattern, self.content, re.DOTALL | re.IGNORECASE)
        return rows
    
    def get_column_count(self) -> int:
        if not self.rows:
            return 0
        import re
        td_count = len(re.findall(r'<td[^>]*>', self.rows[0], re.IGNORECASE))
        th_count = len(re.findall(r'<th[^>]*>', self.rows[0], re.IGNORECASE))
        return max(td_count, th_count)


def can_merge_html_tables(table1: HTMLTableBlock, table2: HTMLTableBlock) -> bool:
    if table2.page_num != table1.page_num + 1:
        return False
    
    if table1.get_column_count() != table2.get_column_count():
        return False
    
    return True


def merge_html_tables(table1: HTMLTableBlock, table2: HTMLTableBlock) -> str:
    import re
    
    table_opening_match = re.match(r'<table[^>]*>', table1.content, re.IGNORECASE)
    if not table_opening_match:
        return table1.content
    
    table1_opening = table_opening_match.group(0)
    rows1 = table1.rows
    rows2 = table2.rows
    all_rows = rows1 + rows2
    
    merged_table = table1_opening + '\n' + '\n'.join(all_rows) + '\n</table>'
    return merged_table


def _parse_html_tables(pages_content: List[str]) -> List[HTMLTableBlock]:
    html_tables = []
    
    for page_num, page_content in enumerate(pages_content):
        tables_in_page = extract_html_tables(page_content)
        
        for start_pos, end_pos, table_content in tables_in_page:
            html_table = HTMLTableBlock(table_content, page_num, start_pos, end_pos)
            html_tables.append(html_table)
    
    return html_tables


def _identify_html_merge_groups(html_tables: List[HTMLTableBlock]) -> List[List[int]]:
    merge_groups = []
    i = 0
    
    while i < len(html_tables):
        current_group = [i]
        
        j = i + 1
        while j < len(html_tables):
            if can_merge_html_tables(html_tables[current_group[-1]], html_tables[j]):
                current_group.append(j)
                j += 1
            else:
                break
        
        if len(current_group) > 1:
            merge_groups.append(current_group)
        
        i = current_group[-1] + 1
    
    return merge_groups


def merge_cross_page_html_tables(pages_content: List[str]) -> List[str]:
    html_tables = _parse_html_tables(pages_content)
    
    if not html_tables:
        return pages_content
    
    merge_groups = _identify_html_merge_groups(html_tables)
    
    if not merge_groups:
        return pages_content
    
    result_pages = []
    
    for page_num, page_content in enumerate(pages_content):
        page_html_tables = [t for t in html_tables if t.page_num == page_num]
        
        if not page_html_tables:
            result_pages.append(page_content)
            continue
        
        modified_content = page_content
        
        for table in sorted(page_html_tables, key=lambda t: t.start_pos, reverse=True):
            table_idx = html_tables.index(table)
            
            merge_group = None
            for group in merge_groups:
                if table_idx in group:
                    merge_group = group
                    break
            
            if merge_group and table_idx == merge_group[0]:
                tables_to_merge = [html_tables[idx] for idx in merge_group]
                merged_table = tables_to_merge[0].content
                
                for next_table in tables_to_merge[1:]:
                    merged_table = merge_html_tables(
                        HTMLTableBlock(merged_table, 0, 0, 0),
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
