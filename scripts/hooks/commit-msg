#!/bin/sh

pattern='^(python|java|feat|fix|docs|style|refactor|perf|test|chore)(\([a-z0-9-]+\))?: .{1,50}$'

commit_msg=$(head -n1 "$1")

if ! echo "$commit_msg" | grep -qE "$pattern"; then
    echo "❌ Commit message format invalid."
    echo ""
    echo "✅ Expected format: <type>(optional-scope): <subject>"
    echo "   Example: feat(auth): add JWT support"
    echo ""
    exit 1
fi

exit 0