/**
 * scripts/autoclose-labeled-issues.js
 *
 * Auto-closes issues that have a bot "possible duplicate" comment older than
 * 3 days, unless:
 * - A human has commented after the bot's duplicate comment
 * - The author reacted with thumbs-down on the duplicate comment
 *
 * Required environment variables:
 *   GITHUB_TOKEN  - GitHub Actions token
 *   REPO_OWNER    - Repository owner
 *   REPO_NAME     - Repository name
 *
 * Optional:
 *   DRY_RUN       - If "true", report but do not close (default: false)
 */

'use strict';

const https = require('https');

const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const REPO_OWNER   = process.env.REPO_OWNER;
const REPO_NAME    = process.env.REPO_NAME;
const DRY_RUN      = process.env.DRY_RUN === 'true';

const THREE_DAYS_MS = 3 * 24 * 60 * 60 * 1000;

function githubRequest(method, path, body = null, retried = false) {
  return new Promise((resolve, reject) => {
    const payload = body ? JSON.stringify(body) : null;
    const options = {
      hostname: 'api.github.com',
      path,
      method,
      headers: {
        'Authorization': `Bearer ${GITHUB_TOKEN}`,
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'PageIndex-Autoclose/1.0',
        'X-GitHub-Api-Version': '2022-11-28',
        ...(payload ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) } : {}),
      },
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', chunk => (data += chunk));
      res.on('end', async () => {
        // 429: 始终重试（rate limit）
        if (res.statusCode === 429 && !retried) {
          const retryAfter = parseInt(res.headers['retry-after'] || '60', 10);
          console.log(`  Rate limited on ${method} ${path}, retrying after ${retryAfter}s...`);
          await sleep(retryAfter * 1000);
          try { resolve(await githubRequest(method, path, body, true)); }
          catch (err) { reject(err); }
          return;
        }
        // 403: 只在 rate limit 相关时重试
        if (res.statusCode === 403 && !retried) {
          const rateLimitRemaining = res.headers['x-ratelimit-remaining'];
          const hasRetryAfter = res.headers['retry-after'];
          if (rateLimitRemaining === '0' || hasRetryAfter) {
            const retryAfter = parseInt(hasRetryAfter || '60', 10);
            console.log(`  Rate limited (403) on ${method} ${path}, retrying after ${retryAfter}s...`);
            await sleep(retryAfter * 1000);
            try { resolve(await githubRequest(method, path, body, true)); }
            catch (err) { reject(err); }
            return;
          }
        }
        if (res.statusCode >= 400) {
          reject(new Error(`GitHub API ${method} ${path} -> ${res.statusCode}: ${data}`));
          return;
        }
        try { resolve(data ? JSON.parse(data) : {}); }
        catch { resolve({}); }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * Fetches open issues with the "duplicate" label, paginating as needed.
 * Only returns issues created more than 3 days ago.
 */
async function fetchDuplicateIssues() {
  const issues = [];
  let page = 1;
  while (true) {
    const data = await githubRequest(
      'GET',
      `/repos/${REPO_OWNER}/${REPO_NAME}/issues?state=open&labels=duplicate&per_page=100&page=${page}`
    );
    if (!Array.isArray(data) || data.length === 0) break;
    issues.push(...data.filter(i => !i.pull_request));
    if (data.length < 100) break;
    page++;
  }

  const cutoff = new Date(Date.now() - THREE_DAYS_MS);
  return issues.filter(i => new Date(i.created_at) < cutoff);
}

function isBot(user) {
  return user.type === 'Bot' || user.login.endsWith('[bot]') || user.login === 'github-actions';
}

/**
 * Finds the bot's duplicate comment on an issue (contains "possible duplicate").
 */
function findDuplicateComment(comments) {
  return comments.find(c =>
    isBot(c.user) && c.body.includes('possible duplicate')
  );
}

/**
 * Checks if there are human comments after the duplicate comment.
 */
function hasHumanCommentAfter(comments, afterDate) {
  return comments.some(c => {
    if (isBot(c.user)) return false;
    return new Date(c.created_at) > afterDate;
  });
}

/**
 * Fetches all comments for an issue, handling pagination.
 * Requests per_page=100 and loops until we get fewer than 100 or an empty array.
 */
async function fetchAllComments(issueNumber) {
  const allComments = [];
  let page = 1;
  while (true) {
    const comments = await githubRequest(
      'GET',
      `/repos/${REPO_OWNER}/${REPO_NAME}/issues/${issueNumber}/comments?per_page=100&page=${page}`
    );
    if (!Array.isArray(comments) || comments.length === 0) break;
    allComments.push(...comments);
    if (comments.length < 100) break;
    page++;
  }
  return allComments;
}

/**
 * Checks if the duplicate comment has a thumbs-down reaction.
 */
async function hasThumbsDownReaction(commentId) {
  const reactions = await githubRequest(
    'GET',
    `/repos/${REPO_OWNER}/${REPO_NAME}/issues/comments/${commentId}/reactions`
  );
  return Array.isArray(reactions) && reactions.some(r => r.content === '-1');
}

/**
 * Closes an issue as duplicate with a comment.
 */
async function closeAsDuplicate(issueNumber) {
  const body =
    'This issue has been automatically closed as a duplicate. ' +
    'No human activity or objection was received within the 3-day grace period.\n\n' +
    'If you believe this was closed in error, please reopen the issue and leave a comment.';

  await githubRequest(
    'POST',
    `/repos/${REPO_OWNER}/${REPO_NAME}/issues/${issueNumber}/comments`,
    { body }
  );

  await githubRequest(
    'PATCH',
    `/repos/${REPO_OWNER}/${REPO_NAME}/issues/${issueNumber}`,
    { state: 'closed', state_reason: 'completed' }
  );
}

async function processIssue(issue) {
  const num = issue.number;
  console.log(`\nChecking issue #${num}: ${issue.title}`);

  const comments = await fetchAllComments(num);

  if (!Array.isArray(comments) || comments.length === 0) {
    console.log(`  -> Could not fetch comments, skipping.`);
    return false;
  }

  const dupeComment = findDuplicateComment(comments);
  if (!dupeComment) {
    console.log(`  -> No duplicate comment found, skipping.`);
    return false;
  }

  const commentDate = new Date(dupeComment.created_at);
  const ageMs = Date.now() - commentDate.getTime();

  if (ageMs < THREE_DAYS_MS) {
    const daysLeft = Math.ceil((THREE_DAYS_MS - ageMs) / (24 * 60 * 60 * 1000));
    console.log(`  -> Duplicate comment is less than 3 days old (${daysLeft}d remaining), skipping.`);
    return false;
  }

  if (hasHumanCommentAfter(comments, commentDate)) {
    console.log(`  -> Human commented after duplicate comment, skipping.`);
    return false;
  }

  if (await hasThumbsDownReaction(dupeComment.id)) {
    console.log(`  -> Author reacted with thumbs-down, skipping.`);
    return false;
  }

  if (DRY_RUN) {
    console.log(`  [DRY RUN] Would close issue #${num}`);
    return true;
  }

  await closeAsDuplicate(num);
  console.log(`  -> Closed issue #${num} as duplicate`);
  return true;
}

async function main() {
  const missing = ['GITHUB_TOKEN', 'REPO_OWNER', 'REPO_NAME'].filter(k => !process.env[k]);
  if (missing.length) {
    console.error(`Missing required environment variables: ${missing.join(', ')}`);
    process.exit(1);
  }

  console.log('Auto-close duplicate issues');
  console.log(`  Repository: ${REPO_OWNER}/${REPO_NAME}`);
  console.log(`  Dry run:    ${DRY_RUN}`);

  const issues = await fetchDuplicateIssues();
  console.log(`\nFound ${issues.length} duplicate-labeled issue(s) older than 3 days.`);

  let closedCount = 0;
  for (const issue of issues) {
    const closed = await processIssue(issue);
    if (closed) closedCount++;
    await sleep(1000);
  }

  console.log(`\nSummary: ${closedCount} issue(s) closed.`);
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
