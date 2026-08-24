import os
import re
import sys
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GITHUB_API_VERSION = "2022-11-28"

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPOSITORY = os.environ["GITHUB_REPOSITORY"]
PR_NUMBER = os.environ["PR_NUMBER"]
HEAD_SHA = os.environ["HEAD_SHA"]
MODEL = os.environ["MODEL"]
DIFF_FILE = os.environ["DIFF_FILE"]
CHUNK_CHAR_BUDGET = int(os.environ.get("CHUNK_CHAR_BUDGET", "12000"))
MAX_COMPLETION_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "1024"))
MAX_COMMENT_SIZE = int(os.environ.get("MAX_COMMENT_SIZE", "60000"))
EXTRA_INSTRUCTIONS = os.environ.get("EXTRA_INSTRUCTIONS", "").strip()
COMMENT_MARKER = os.environ.get("COMMENT_MARKER", "<!-- groq-code-review -->")
FAIL_ON_REVIEW_ERROR = os.environ.get("FAIL_ON_REVIEW_ERROR", "false").lower() == "true"

_EXTRA_BLOCK = f"\n{EXTRA_INSTRUCTIONS}" if EXTRA_INSTRUCTIONS else ""

SYSTEM_PROMPT = f"""You are an experienced senior software engineer reviewing a GitHub pull request.

Review the supplied code diff carefully.

Focus on:
- Bugs and incorrect behavior
- Security vulnerabilities
- Infrastructure/IaC risks
- AWS/CDK mistakes
- IAM permission problems
- Reliability and availability issues
- Breaking changes
- Performance problems
- Error handling
- Maintainability

Do NOT complain about formatting unless it causes a real problem.

Be concise and actionable.

For every important finding include:
1. Severity: CRITICAL, HIGH, MEDIUM, or LOW
2. File/path when identifiable
3. What is wrong
4. How to fix it

If there are no meaningful issues, say so clearly.
{_EXTRA_BLOCK}"""


def split_into_file_diffs(diff_text):
    """Split a unified diff on file boundaries so a chunk never cuts a file in half."""
    parts = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)
    return [p for p in parts if p.strip()]


def pack_chunks(file_diffs, budget):
    """Pack whole file-diffs into chunks up to `budget` chars. A single file
    larger than the budget is sliced on its own so nothing is silently dropped."""
    chunks = []
    current, current_len = [], 0

    def flush():
        if current:
            chunks.append("".join(current))
            current.clear()

    for fd in file_diffs:
        if len(fd) > budget:
            flush()
            current_len = 0
            for i in range(0, len(fd), budget):
                chunks.append(fd[i:i + budget])
            continue
        if current_len + len(fd) > budget and current:
            flush()
            current_len = 0
        current.append(fd)
        current_len += len(fd)
    flush()
    return chunks


def call_groq(chunk, index, total, max_retries=3):
    user_prompt = (
        f"Review chunk {index} of {total} of this pull request.\n\n"
        f"```diff\n{chunk}\n```\n\n"
        "Return only actionable review findings."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": "low",
    }

    delay = 5
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
        except requests.RequestException as exc:
            last_err = f"request failed: {exc}"
            print(f"Chunk {index}/{total} attempt {attempt}: {last_err}")
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 429:
            last_err = "rate limited (429)"
            print(f"Chunk {index}/{total} attempt {attempt}: {last_err}, backing off {delay}s")
            time.sleep(delay)
            delay *= 2
            continue

        if not resp.ok:
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            print(f"Chunk {index}/{total} attempt {attempt}: {last_err}")
            time.sleep(delay)
            delay *= 2
            continue

        content = resp.json()["choices"][0]["message"]["content"]
        return content, None

    return None, last_err


def find_existing_comment(headers):
    url = f"https://api.github.com/repos/{REPOSITORY}/issues/{PR_NUMBER}/comments"
    page = 1
    while True:
        resp = requests.get(url, headers=headers, params={"per_page": 100, "page": page}, timeout=60)
        resp.raise_for_status()
        comments = resp.json()
        if not comments:
            return None
        for c in comments:
            if COMMENT_MARKER in c.get("body", ""):
                return c["id"]
        if len(comments) < 100:
            return None
        page += 1


def post_or_update_comment(body):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    existing_id = find_existing_comment(headers)

    if existing_id:
        url = f"https://api.github.com/repos/{REPOSITORY}/issues/comments/{existing_id}"
        resp = requests.patch(url, headers=headers, json={"body": body}, timeout=60)
    else:
        url = f"https://api.github.com/repos/{REPOSITORY}/issues/{PR_NUMBER}/comments"
        resp = requests.post(url, headers=headers, json={"body": body}, timeout=60)

    if not resp.ok:
        print("GitHub API error:")
        print(resp.text)
        resp.raise_for_status()


def main():
    with open(DIFF_FILE, "r", encoding="utf-8", errors="replace") as f:
        diff = f.read()

    if not diff.strip():
        body = (
            f"{COMMENT_MARKER}\n"
            "## Groq AI Code Review\n\n"
            "No reviewable changes found in this diff.\n"
        )
        post_or_update_comment(body)
        print("No diff content; posted a no-op review.")
        return

    file_diffs = split_into_file_diffs(diff)
    chunks = pack_chunks(file_diffs, CHUNK_CHAR_BUDGET)

    print(f"Diff size: {len(diff)} characters")
    print(f"Files changed: {len(file_diffs)}")
    print(f"Review chunks: {len(chunks)}")

    reviews = []
    failures = 0

    for index, chunk in enumerate(chunks, start=1):
        print(f"Reviewing chunk {index}/{len(chunks)}...")
        content, err = call_groq(chunk, index, len(chunks))

        if err is not None:
            failures += 1
            reviews.append(f"### Chunk {index}/{len(chunks)}\n\n_Review failed: {err}_")
        elif content and content.strip():
            reviews.append(f"### Chunk {index}/{len(chunks)}\n\n{content.strip()}")

        if index < len(chunks):
            time.sleep(1.5)

    if reviews:
        review_body = "\n\n---\n\n".join(reviews)
    else:
        review_body = "No significant issues were found in the reviewed changes."

    status_line = ""
    if failures:
        status_line = f"\n> ⚠️ {failures}/{len(chunks)} chunk(s) could not be reviewed. See details below.\n"

    comment = f"""{COMMENT_MARKER}
## Groq AI Code Review

**Model:** `{MODEL}` · **Reviewed commit:** `{HEAD_SHA[:7]}`
{status_line}
{review_body}

---
_Automated review. Treat as a second opinion, not a substitute for human review or IaC-specific tooling (e.g. cdk-nag, checkov)._
"""

    if len(comment) > MAX_COMMENT_SIZE:
        comment = comment[:MAX_COMMENT_SIZE] + "\n\n> Review truncated because the GitHub comment was too large."

    post_or_update_comment(comment)
    print("Groq review posted successfully.")

    if failures and FAIL_ON_REVIEW_ERROR:
        print(f"::error::{failures}/{len(chunks)} chunks failed to review.")
        sys.exit(1)


if __name__ == "__main__":
    main()
