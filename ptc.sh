#!/bin/bash

# Usage: ./ptc.sh <file_path> [options]
# Example: ./ptc.sh /path/to/data -i "*.ply" -o -f npy -c 100 -d 50

# Ensure at least one argument (file_path) is provided
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <file_path> [-i <include_pattern>] [-o] [-f <format>] [-c <crop>] [-d <depth>]"
    exit 1
fi

# 1️⃣ Assign variables and initialize defaults
FILE_PATH=""
INCLUDE_PATTERN=""
OVERWRITE_FLAG=""
FORMAT="ply"  # Default format
CROP_SIZE=112  # Default crop size (mm)
MAX_DEPTH=56  # Default depth (mm)

# 2️⃣ Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -i|--include)
            INCLUDE_PATTERN="$2"
            shift 2
            ;;
        -o|--overwrite)
            OVERWRITE_FLAG="--overwrite"
            shift
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        -c|--crop)
            CROP_SIZE="$2"
            shift 2
            ;;
        -d|--depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        *)
            if [[ -z "$FILE_PATH" ]]; then
                FILE_PATH="$1"
                shift
            else
                echo "❌ Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

# 3️⃣ Convert relative FILE_PATH to absolute path
ABS_FILE_PATH="$(realpath "$FILE_PATH")"

# 4️⃣ Validate the resolved path
if [[ ! -e "$ABS_FILE_PATH" ]]; then
    echo "❌ Error: Path '$ABS_FILE_PATH' does not exist."
    exit 1
fi

# 5️⃣ Determine volume mount point
if [[ -f "$ABS_FILE_PATH" ]]; then
    # If it's a file, mount its parent directory
    MOUNT_DIR="$(dirname "$ABS_FILE_PATH")"
    CONTAINER_PATH="/mnt/$(basename "$ABS_FILE_PATH")"
elif [[ -d "$ABS_FILE_PATH" ]]; then
    # If it's a directory, mount the entire directory
    MOUNT_DIR="$ABS_FILE_PATH"
    CONTAINER_PATH="/mnt"
else
    echo "❌ Error: '$ABS_FILE_PATH' is neither a valid file nor directory."
    exit 1
fi

# 5️⃣ Run the Docker container with the correct volume and pass arguments
docker run --rm -v "$MOUNT_DIR:/mnt" rec-utils:latest \
    python ptc.py "$CONTAINER_PATH" \
    ${INCLUDE_PATTERN:+--include "$INCLUDE_PATTERN"} \
    $OVERWRITE_FLAG \
    --format "$FORMAT" \
    --crop "$CROP_SIZE" \
    --depth "$MAX_DEPTH"