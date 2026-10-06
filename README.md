# Groq AI Code Review

A GitHub Action that posts an AI-generated pull request review using the [Groq API](https://groq.com/), tuned to catch bugs, security issues, and infrastructure-as-code / AWS CDK / IAM mistakes.

It diffs the PR against its base branch, splits the diff on file boundaries (so a file is never reviewed as a broken fragment), sends each chunk to a Groq model, and posts a single PR comment — updating that same comment on later pushes instead of spamming new ones.

This is a **second opinion**, not a merge gate. It doesn't approve, block, or auto-merge anything — it only comments.

## Usage

```yaml
name: AI Code Review

on:
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened]
    paths-ignore:
      - "**/*.md"
      - "LICENSE"

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: GudditiN/groq-cdk-code-review@v1
        with:
          groq-api-key: ${{ secrets.GROQ_API_KEY }}
```

The action fetches the exact base/head commits it needs by SHA, so a plain `actions/checkout@v4` (default shallow depth) is enough — you do **not** need `fetch-depth: 0`.

### Setting up `GROQ_API_KEY`

1. Get an API key from [console.groq.com](https://console.groq.com/keys).
2. On the repo that will *use* this action (not necessarily this repo): **Settings → Secrets and variables → Actions → New repository secret**.
3. Name it `GROQ_API_KEY` and paste the key as the value.
4. Reference it in the workflow as `groq-api-key: ${{ secrets.GROQ_API_KEY }}`, as in the example above.

If the key is missing, invalid, or the configured `model` is wrong/inaccessible, the action fails fast with a clear `::error::` message (and posts an explanatory PR comment) instead of silently retrying or producing an empty review.

### Required GitHub permissions

Set these at the workflow or job level (as in the example):

| Permission | Why |
|---|---|
| `contents: read` | Needed for `actions/checkout` and for the action's internal `git fetch`/`git diff` against the base and head commits. |
| `pull-requests: write` | Needed to post the review comment, and to read/update it on later pushes (via the issue-comments API, which pull requests share). |

Without `pull-requests: write`, the GitHub API calls to create/update the comment will fail with a 403.

## Inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `groq-api-key` | yes | — | Groq API key. |
| `github-token` | no | `${{ github.token }}` | Token used to read/post PR comments. |
| `model` | no | `openai/gpt-oss-120b` | Groq model id. |
| `chunk-char-budget` | no | `12000` | Approx. max chars of diff per request. See "How chunking works" below. |
| `max-completion-tokens` | no | `1024` | Max tokens per chunk response. |
| `max-comment-size` | no | `60000` | Max characters of the final PR comment before truncation. |
| `max-chunks` | no | `40` | Max number of chunks reviewed per PR. Extra chunks are skipped (noted in the comment) rather than sending an unbounded number of requests on a huge PR. |
| `temperature` | no | `0.2` | Sampling temperature (0–2) sent to the Groq API. Lower is more consistent/deterministic, which is generally preferable for a reviewer. |
| `exclude-paths` | no | binaries, lockfiles, snapshots, `cdk.out/**` | Newline-separated git pathspec excludes. |
| `context-char-budget` | no | `24000` | Max chars of repository context sent with each chunk. `0` sends only the diff. See "Repository context" below. |
| `extra-instructions` | no | `''` | Extra text appended to the reviewer's system prompt (e.g. house rules). |
| `comment-marker` | no | `<!-- groq-code-review -->` | Hidden marker used to find and update a previous review comment. |
| `fail-on-review-error` | no | `false` | Fail the job if any chunk couldn't be reviewed after retries. |

### How chunking works

The diff is split on `diff --git` boundaries first, so a chunk never cuts a file's hunk in half. Whole files are then packed together up to `chunk-char-budget` characters per request. If a single file's diff is *larger* than the budget on its own (common for generated files, large snapshots, or big lockfiles that slipped past `exclude-paths`), it's sliced into consecutive sub-chunks and reviewed across multiple requests — the model only sees a fragment of that file per request in that case, so review quality degrades for files that big. Prefer adding such files to `exclude-paths` over relying on slicing.

If the PR produces more chunks than `max-chunks`, only the first `max-chunks` are reviewed and the comment notes how many were skipped.

### Repository context

A diff alone can't show how a change fits the rest of the codebase. For example, it doesn't show which stack consumes a prop, where a `deployment-config` key is read, or whether a construct already sets a default. So with each chunk the action also sends context read from the PR's head commit (via `git show` / `git grep`, no extra checkout needed):

1. **Repository layout**: the source files grouped by directory. Compiled `.js`/`.d.ts` next to `.ts` sources, `node_modules`, `cdk.out` and lockfiles are skipped.
2. **Changed files**: each changed file in full when small, otherwise the hunks with surrounding lines, with line numbers.
3. **References**: identifiers added or removed in the diff (env var names, config keys, class/stack names, kebab-case resource ids) are searched across the repo, and the matching lines are included. That is how the model sees, for example, that `config.shopDb.minCapacityAcu` is passed in from `bin/infra.ts`, or that `additionalBehaviors` is forwarded by a shared construct. Identifiers that appear in too many files are skipped as too generic.

About 40% of the budget is reserved for references. The model is told to use this context as evidence, to report only problems in the diff, and not to speculate about consumers outside the repository. The context comes from the PR and is treated as untrusted, in the same way as the diff.

Context makes each request larger (24,000 chars is roughly 6k tokens). If you hit Groq rate limits, lower `context-char-budget`. On HTTP 429 the action waits for Groq's `retry-after` header (up to 60s) before retrying.

## Notes

- **Fork PRs and secrets**: this action runs on the `pull_request` event (not `pull_request_target`) intentionally. For a *public* repository, GitHub does not pass repository secrets — including `GROQ_API_KEY` — to workflow runs triggered by a pull request from a forked repo; only `GITHUB_TOKEN` is available, and it's read-only unless the repo owner has explicitly enabled write tokens for fork PRs (off by default). In practice this means: on a public repo, PRs from forks will fail this action's input validation (missing `GROQ_API_KEY`) rather than leak the key or silently do nothing. If your repo is private, forks generally aren't a factor since only collaborators can open PRs. Do not switch this to `pull_request_target` to "fix" that — it would remove the protection and expose secrets to untrusted PR code.
- The diff content is attacker/contributor-controlled input to the LLM prompt. The system prompt instructs the model to treat the diff strictly as content to analyze (not instructions to follow) and wraps it in a random per-run boundary token so a PR can't forge a matching delimiter — this reduces, but does not eliminate, prompt-injection risk. Don't wire this action's output into auto-merge or required status checks — treat it as an aid for human reviewers, not a gate.
- Groq's free tier has request/token rate limits. The action retries only transient failures (429 rate limits, 5xx server errors, network errors) with exponential backoff; non-transient errors (e.g. a malformed request) are recorded as a per-chunk failure without wasting retries, and auth/model errors (401/403/404 — bad key, bad/inaccessible model) abort the whole run immediately with one clear error instead of repeating the same failure across every chunk. A partial review posts (with failures flagged) rather than failing the whole job, unless `fail-on-review-error: true`.

## License

MIT — see [LICENSE](./LICENSE).
