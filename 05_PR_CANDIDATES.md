# 05_PR_CANDIDATES.md — PageIndex PR Candidate Analysis

## Open Issues with PRs (Active Contribution Opportunity)
These issues have open PRs — check if they are close to mergeable or need help:

| # | Title | PR Branch | Status |
|---|-------|-----------|--------|
| 285 | Route TOC-without-page-numbers documents to the correct strategy | `fix/toc-no-page-numbers-routing` | Open |
| 282 | docs: document additional run_pageindex CLI options | `mack/pr-20260519-1648-pageindex` | Open |
| 281 | Adds missing re import | `patch-1` | Open |
| 280 | fix: return only requested pages from get_page_content on Markdown | `fix/md-discrete-pages-overcollection` | Open |
| 279 | get_page_content over-collects on Markdown when given a comma-separated page list | — | No PR yet, related to #280 |
| 277 | fix: graceful error recovery in TOC builder | `fix/graceful-error-recovery-in-toc-builder` | Open |
| 276 | Feat/fastapi server | `feat/fastapi-server` | Open |
| 274 | fix: add missing commas in LLM prompt JSON formats and guard list return types | `fix/llm-response-robustness` | Open |
| 268 | CLI output path is hardcoded, making scripted usage difficult | `feat/cli-output-path` | Open |
| 267 | Defend against single-page TOC misclassification dropping all content | `fix/single-page-toc-misclassification` | Open |
| 266 | Improve indexing robustness and logging | `fix/robustness-indexing` | Open |
| 264 | fix: clarify single-page toc detection | `codex/single-page-toc-guard` | Open |
| 250 | Recognize whole-line bold as level-1 heading in markdown parser | `feat/md-bold-heading-recognition` | Open |
| 249 | fix: fix markdown parser edge cases in page_index_md.py | `fix/issue-245-markdown-parser-edge-cases` | Open |
| 246 | fix: make markdown parser robust | `markdown_fix` | Open |
| 245 | Markdown parser fails on common edge cases | — | No PR yet |
| 218 | fix: comprehensive crash guards for malformed LLM output | `fix/comprehensive-crash-guards` | Open |
| 213 | fix: apply regex-based trailing comma removal before JSON parse | `fix/issue-195-extract-json-regex-cleanup` | Open |

---

## High-Value Good First Issues (No PR Yet)

### 1. Issue #286 — Requirements Version Conflict
**Title**: "Installation - Requirements with version of litellm need python-dotenv==1.0.1 but conflict with requirements python-dotenv==1.2.2"
- **Problem**: Dependency conflict between litellm (expects python-dotenv==1.0.1) and requirements.txt (has python-dotenv==1.2.2)
- **Impact**: Installation failure for new users
- **Fix**: Pin compatible versions or update litellm to a version that accepts newer dotenv
- **Complexity**: Low — dependency version resolution

### 2. Issue #283 — Unthrottled Concurrent LLM Requests Cause 429 Rate Limits
**Title**: "[Bug] Unthrottled concurrent LLM requests lead to HTTP 429 Rate Limits and cascading KeyError in tree generation"
- **Problem**: Async code fires unlimited concurrent requests, hitting rate limits
- **Impact**: Production reliability, cascading failures
- **Fix**: Add a semaphore/throttle to limit concurrent LLM calls (e.g., `asyncio.Semaphore(5)`)
- **Complexity**: Medium — async concurrency control

### 3. Issue #284 — Multi-Document Literature Review Support
**Title**: "Does PageIndex work well for synthesizing a literature review or survey report from dozens of separate documents"
- **Problem**: Question about multi-document support
- **Opportunity**: Could close with docs update or reference to `feat/multi-doc-support` PR #216
- **Complexity**: Documentation or feature

### 4. Issue #278 — Local Private Model Configuration
**Title**: "如何配置本地私有模型？本地运行是否有像dashboard一样的可视化界面" (Chinese)
- **Problem**: User asking about local/private LLM model setup
- **Fix**: Update docs for LiteLLM local endpoint configuration
- **Complexity**: Low — documentation

---

## Closed/Recent Merged PRs — Pattern Analysis

| PR | Title | Theme |
|----|-------|-------|
| 271 | Update README | Maintenance |
| 262/261/259 | update README | Maintenance |
| 256/255 | Fix Agentic RAG entry formatting | Bug fix |
| 248 | Add security CI workflows | CI/CD |
| 247 | Bump pip group | Dependencies |
| 241 | Add Dependabot config for GitHub Actions | CI/CD |
| 238 | feat: compatible with PageIndex SDK | Feature |
| 228 | feat: Universal Local LLM & Custom Endpoint Support | Feature |
| 227 | feat: add checkpoint/resume for long document processing | Feature |
| 226 | fix: poll status=="completed" in cloud add_document | Bug fix |
| 221 | Add FastAPI server for PageIndex document indexing service | Feature |
| 218 | fix: comprehensive crash guards for malformed LLM output | Bug fix |
| 216 | feat: add multi-document support to retrieval and client API | Feature |
| 213 | fix: apply regex-based trailing comma removal before JSON parse | Bug fix |
| 207 | feat: add PageIndex SDK with local/cloud dual-mode support | Feature |

**Pattern**: Recent work focuses on SDK features, robustness, and multi-document support.

---

## Active Feature Branches in Upstream

| Branch | Description |
|--------|-------------|
| `dev` | Development branch |
| `feat/markdown-tree` | Markdown tree feature |
| `feat/md-bold-heading-recognition` | Bold heading detection |
| `fix/cloud-poll-status-completed` | Cloud polling fix |
| `add-pypdfium2-parser` | Optional pypdfium2 PDF parser |
| `dependabot/*` | Dependency updates (3 branches) |

---

## PR Merge Velocity
- Last 30 closed PRs (not all merged)
- Merged: ~20 PRs in recent weeks
- Active maintenance with good review turnaround

---

## Dependency PRs (Quick Wins)

Three dependabot PRs waiting to merge:
- #275: `actions/dependency-review-action` 4→5
- #243: `actions/github-script` 7→9  
- #242: `actions/checkout` 4→6

These are routine and should be merged.

---

## Recommended Focus Areas for Contribution

1. **Bug fixes with existing PRs**: Issues #280, #274, #249, #246 — PRs exist, review/test them
2. **Markdown parser**: Issues #245, #250, #249, #246 — multiple related issues, could be consolidated
3. **Rate limiting**: Issue #283 — needs implementation of semaphore-based concurrency control
4. **Dependency conflicts**: Issue #286 — straightforward version pinning
5. **Test coverage**: Issue #263 — "100% Unit Test Coverage" — major gap in the project