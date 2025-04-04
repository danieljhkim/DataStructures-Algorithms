#!/bin/bash

HOOK_NAME="commit-msg"
SOURCE_FILE="./scripts/hooks/$HOOK_NAME"
DEST_FILE=".git/hooks/$HOOK_NAME"

if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ Hook script not found at $SOURCE_FILE"
    exit 1
fi

cp "$SOURCE_FILE" "$DEST_FILE"
chmod +x "$DEST_FILE"

echo "✅ Git hook '$HOOK_NAME' installed successfully."