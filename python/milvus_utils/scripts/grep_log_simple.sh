#!/bin/bash

# Simple script to grep a log file every second for specific strings
# Usage: ./grep_log_simple.sh <log_file> <search_string1> [search_string2] [search_string3] ...
#   or set MIXCOORD_LOG_FILE environment variable and use: ./grep_log_simple.sh <search_string1> [search_string2] ...

# Don't use set -e since we need to check grep exit codes

# Get log file from first argument or environment variable
# Note: bash variables can't have hyphens, so using MIXCOORD_LOG_FILE
if [ -n "${MIXCOORD_LOG_FILE}" ]; then
    # If env var is set, all arguments are search strings
    LOG_FILE="${MIXCOORD_LOG_FILE}"
    SEARCH_STRINGS=("$@")
else
    # First argument is log file, rest are search strings
    LOG_FILE="${1}"
    shift
    SEARCH_STRINGS=("$@")
fi

if [ -z "$LOG_FILE" ]; then
    echo "Error: Log file not specified"
    echo "Usage: $0 <log_file> <search_string1> [search_string2] [search_string3] ..."
    echo "   or: MIXCOORD_LOG_FILE=<log_file> $0 <search_string1> [search_string2] ..."
    exit 1
fi

if [ ${#SEARCH_STRINGS[@]} -eq 0 ]; then
    echo "Error: No search strings specified"
    echo "Usage: $0 <log_file> <search_string1> [search_string2] [search_string3] ..."
    echo "   or: MIXCOORD_LOG_FILE=<log_file> $0 <search_string1> [search_string2] ..."
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file '$LOG_FILE' does not exist"
    exit 1
fi

echo "Monitoring log file: $LOG_FILE"
echo "Searching for: ${SEARCH_STRINGS[*]}"
echo "Press Ctrl+C to stop"
echo "----------------------------------------"

# Loop every second, grepping the entire file
# Exit when any string is found
while true; do
    for search_string in "${SEARCH_STRINGS[@]}"; do
        if grep --color=always "$search_string" "$LOG_FILE" 2>/dev/null; then
            echo "String '$search_string' found! Exiting..."
            exit 0
        fi
    done
    sleep 1
done
