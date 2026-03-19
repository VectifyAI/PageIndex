#!/usr/bin/env bash
#
# comment-on-duplicates.sh - Posts a duplicate issue comment with auto-close warning.
#
# Usage:
#   ./scripts/comment-on-duplicates.sh --base-issue 123 --potential-duplicates 456 789
#
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-}"
if [ -z "$REPO" ]; then
  echo "Error: GITHUB_REPOSITORY is not set" >&2
  exit 1
fi

BASE_ISSUE=""
DUPLICATES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-issue)
      BASE_ISSUE="$2"
      shift 2
      ;;
    --potential-duplicates)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        DUPLICATES+=("$1")
        shift
      done
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# Validate inputs
if [ -z "$BASE_ISSUE" ]; then
  echo "Error: --base-issue is required" >&2
  exit 1
fi

if ! [[ "$BASE_ISSUE" =~ ^[0-9]+$ ]]; then
  echo "Error: --base-issue must be a number, got: $BASE_ISSUE" >&2
  exit 1
fi

if [ ${#DUPLICATES[@]} -eq 0 ]; then
  echo "Error: --potential-duplicates requires at least one issue number" >&2
  exit 1
fi

for dup in "${DUPLICATES[@]}"; do
  if ! [[ "$dup" =~ ^[0-9]+$ ]]; then
    echo "Error: duplicate issue must be a number, got: $dup" >&2
    exit 1
  fi
done

# Limit to 3 duplicates max
if [ ${#DUPLICATES[@]} -gt 3 ]; then
  echo "Warning: Limiting to first 3 duplicates" >&2
  DUPLICATES=("${DUPLICATES[@]:0:3}")
fi

# Build the duplicate links list
COUNT=0
LINKS=""
for dup in "${DUPLICATES[@]}"; do
  COUNT=$((COUNT + 1))
  LINKS="${LINKS}${COUNT}. https://github.com/${REPO}/issues/${dup}
"
done

# Build and post the comment — if the issue is closed or doesn't exist, gh will error out
COMMENT="Found ${COUNT} possible duplicate issue(s):

${LINKS}
This issue will be automatically closed as a duplicate in 3 days.
- To prevent auto-closure, add a comment or react with :thumbsdown: on this comment."

gh issue comment "$BASE_ISSUE" --repo "$REPO" --body "$COMMENT"
gh issue edit "$BASE_ISSUE" --repo "$REPO" --add-label "duplicate"

echo "Posted duplicate comment on issue #$BASE_ISSUE with $COUNT potential duplicate(s)"
