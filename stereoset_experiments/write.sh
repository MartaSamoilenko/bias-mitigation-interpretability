#!/bin/bash

# Output file
OUTPUT_FILE="out.txt"

# Clear or create the output file
> "$OUTPUT_FILE"

echo "Collecting .py files into $OUTPUT_FILE..."

# Find all .py files in current directory and subdirectories
# -print0 and read -d '' handle filenames with spaces correctly
find . -type f -name "*.py" -print0 | while IFS= read -r -d '' file; do
    echo "Processing: $file"

    # Write a separator and the filename to the output file
    echo "==========================================" >> "$OUTPUT_FILE"
    echo "FILE: $file" >> "$OUTPUT_FILE"
    echo "==========================================" >> "$OUTPUT_FILE"

    # Append the file content
    cat "$file" >> "$OUTPUT_FILE"

    # Add a couple of newlines for separation
    echo -e "\n\n" >> "$OUTPUT_FILE"
done

echo "Done! All content saved to $OUTPUT_FILE"