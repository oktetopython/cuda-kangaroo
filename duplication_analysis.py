
import sys
import hashlib
from collections import defaultdict

def get_code_blocks(file_path, window_size=10):
    """
    Yields code blocks from a file using a sliding window.
    Strips whitespace and ignores comments.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return

    processed_lines = []
    original_line_numbers = []
    for i, line in enumerate(lines):
        # Basic cleaning: strip whitespace and remove common single-line comments
        cleaned_line = line.split('//')[0].split('#')[0].strip()
        if cleaned_line:
            processed_lines.append(cleaned_line)
            original_line_numbers.append(i + 1)

    if len(processed_lines) < window_size:
        return

    for i in range(len(processed_lines) - window_size + 1):
        block_lines = processed_lines[i:i + window_size]
        start_line = original_line_numbers[i]
        end_line = original_line_numbers[i + window_size - 1]
        block_content = "\n".join(block_lines)
        yield (start_line, end_line, block_content)

def main():
    """
    Main function to perform code duplication analysis.
    """
    window_size = 10
    hashes = defaultdict(list)

    for file_path in sys.stdin:
        file_path = file_path.strip()
        if not file_path:
            continue

        for start_line, end_line, block in get_code_blocks(file_path, window_size):
            block_hash = hashlib.sha256(block.encode('utf-8')).hexdigest()
            hashes[block_hash].append({
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "block": block
            })

    # Generate Markdown report
    print("# H1: Code Duplication Analysis Report")
    print("\n---\n")

    duplicates_found = False
    for block_hash, locations in hashes.items():
        if len(locations) > 1:
            duplicates_found = True
            print(f"## H2: Duplicated Block (SHA256: `{block_hash[:12]}...`)")
            # Using triple backticks for the code block
            print(f"\n```\n{locations[0]['block']}\n```\n")
            print("### H3: Locations")
            for loc in locations:
                print(f"- **File**: `{loc['file_path']}` (Lines: {loc['start_line']}-{loc['end_line']})")
            print("\n---\n")

    if not duplicates_found:
        print("No code duplication found across the analyzed files.")

if __name__ == "__main__":
    main()
