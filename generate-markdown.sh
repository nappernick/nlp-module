#!/bin/bash

OUTPUT_FILE="python_files.md"
echo "# Python Files" > $OUTPUT_FILE  # Start the output file with a header

find . -type f -name "*.py" \
   ! -path "*/node_modules/*" \
   ! -path "*/.*/*" \
   ! -path "*/dist/*" \
   ! -path "*/build/*" \
   ! -path "*/.git/*" \
   ! -path "*/logs/*" \
   ! -path "*/coverage/*" \
   ! -name "jest.config.js" \
   ! -name "webpack.config.js" \
   ! -name "tsconfig.json" \
   ! -name "bunfig.toml" \
   ! -name "package.json" \
   ! -name "package-lock.json" \
   ! -name "bun.lockb" \
   | sort \
   | while read file; do
       echo -e "\n## $file\n" >> $OUTPUT_FILE
       echo '```python' >> $OUTPUT_FILE
       cat "$file" >> $OUTPUT_FILE
       echo -e '\n```\n' >> $OUTPUT_FILE
   done

echo "Markdown file generated: $OUTPUT_FILE"