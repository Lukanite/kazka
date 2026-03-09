import json
import sys

def print_conversations_from_jsonl(file_path):
    """Read a JSONL file and print user_input and assistant_response pairs."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                try:
                    entry = json.loads(line)
                    
                    # Extract fields
                    user_input = entry.get('user_input', '')
                    assistant_response = entry.get('assistant_response', '')
                    
                    # Print with separators
                    print(f"=== Entry {line_num} ===")
                    print(f"User: {user_input}")
                    print(f"Assistant: {assistant_response}")
                    print()
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_jsonl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print_conversations_from_jsonl(file_path)
