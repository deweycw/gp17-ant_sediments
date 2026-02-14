#!/bin/bash
# List filenames in a directory, quoted and comma-separated, one per line

dir="${1:-.}"
output="${2:-filelist.txt}"

ls -1 "$dir" | sed 's/.*/"&",/' > "$output"

echo "Saved file list to $output"
