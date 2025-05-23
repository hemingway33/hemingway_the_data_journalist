import os

def list_leaf_files(directory):
    """List all leaf file names under the given directory, grouped by file type."""
    file_groups = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip if there are subdirectories
        if dirs:
            continue
            
        # Add all files in this directory, grouped by extension
        for file in files:
            # Get file extension (or 'no_extension' if none)
            ext = os.path.splitext(file)[1].lower()
            if not ext:
                ext = 'no_extension'
                
            # Initialize list for this extension if needed
            if ext not in file_groups:
                file_groups[ext] = []
                
            file_groups[ext].append(file)
            
    return file_groups

# Directory path
directory = "/Users/hemingway/Documents/LLS-archive/企业微信Caches/Files"

# Get and print leaf files
leaf_files = list_leaf_files(directory)

# Save to text file
with open('leaf_files.txt', 'w', encoding='utf-8') as f:
    f.write("Leaf files found:\n")
    for ext, files in leaf_files.items():
        f.write(f"\n{ext} files:\n")
        for file in files:
            f.write(f"- {file}\n")
