#!/bin/bash

# Ensure at least one argument (subcommand) is provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <subcommand> <file_path> -e <engine> [-i <include_pattern>] [-c <crop_size>] [-d <max_depth>] [-o]"
    echo "Subcommands: create_model_input, extract_templates, compare_templates, rank1"
    exit 1
fi

# 1️⃣ Assign variables
SUBCOMMAND=""
FILE_PATH=""
ENGINE=""
INCLUDE_PATTERN=""
CROP_SIZE=112  # Default crop size (mm)
MAX_DEPTH=56   # Default max depth (mm)
OVERWRITE_FLAG=""

# 2️⃣ Parse command-line arguments
SUBCOMMAND="$1"
shift

FILE_PATH="$1"
shift

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        create_model_input|extract_templates|compare_templates|rank1)
            SUBCOMMAND="$1"
            shift
            ;;
        -e|--engine)
            ENGINE="$2"
            shift 2
            ;;
        -i|--include)
            INCLUDE_PATTERN="$2"
            shift 2
            ;;
        -c|--crop_size)
            CROP_SIZE="$2"
            shift 2
            ;;
        -d|--max_depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        -o|--overwrite)
            OVERWRITE_FLAG="--overwrite"
            shift
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

# 3️⃣ Validate required arguments
if [[ -z "$SUBCOMMAND" ]]; then
    echo "❌ Error: Subcommand is required. Choose from: create_model_input, extract_templates, compare_templates, rank1"
    exit 1
fi

if [[ -z "$FILE_PATH" ]]; then
    echo "❌ Error: File path is required."
    exit 1
fi

if [[ -z "$ENGINE" ]]; then
    echo "❌ Error: Engine (-e) is required."
    exit 1
fi

# 4️⃣ Convert relative FILE_PATH to absolute path
ABS_FILE_PATH="$(realpath "$FILE_PATH")"

# 5️⃣ Validate the resolved path
if [[ ! -e "$ABS_FILE_PATH" ]]; then
    echo "❌ Error: Path '$ABS_FILE_PATH' does not exist."
    exit 1
fi

# 6️⃣ Define the Docker command string
docker run --rm -v "$(dirname "$ABS_FILE_PATH"):/mnt" rec-utils:latest \
    python recognition.py "$SUBCOMMAND" \
    --engine "$ENGINE" \
    ${INCLUDE_PATTERN:+--include "$INCLUDE_PATTERN"} \
    --crop_size "$CROP_SIZE" \
    --max_depth "$MAX_DEPTH" \
    $OVERWRITE_FLAG \
    "/mnt/$(basename "$ABS_FILE_PATH")"

