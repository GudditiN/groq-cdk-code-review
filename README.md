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
        with:
          fetch-depth: 0

      - uses: GudditiN/groq-cdk-code-review@v1
        with:
          groq-api-key: ${{ secrets.GROQ_API_KEY }}
```

`fetch-depth: 0` is required so the action can diff against the PR's base commit. You need a `GROQ_API_KEY` secret set on the *consuming* repo (Settings → Secrets and variables → Actions).

## Inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `groq-api-key` | yes | — | Groq API key. |
| `github-token` | no | `${{ github.token }}` | Token used to read/post PR comments. |
| `model` | no | `openai/gpt-oss-20b` | Groq model id. |
| `chunk-char-budget` | no | `12000` | Approx. max chars of diff per request. Whole files are packed together up to this budget; an oversized single file is sliced on its own. |
| `max-completion-tokens` | no | `1024` | Max tokens per chunk response. |
| `max-comment-size` | no | `60000` | Max characters of the final PR comment before truncation. |
| `exclude-paths` | no | binaries, lockfiles, snapshots, `cdk.out/**` | Newline-separated git pathspec excludes. |
| `extra-instructions` | no | `''` | Extra text appended to the reviewer's system prompt (e.g. house rules). |
| `comment-marker` | no | `<!-- groq-code-review -->` | Hidden marker used to find and update a previous review comment. |
| `fail-on-review-error` | no | `false` | Fail the job if any chunk couldn't be reviewed after retries. |

## Notes

- Runs on the `pull_request` event (not `pull_request_target`), so on public repos, secrets are not exposed to workflow runs triggered from forked PRs — only `GITHUB_TOKEN` is (read-only, unless the repo explicitly opts fork PRs into write tokens).
- The diff content is attacker/contributor-controlled input to the LLM prompt. Don't wire this action's output into auto-merge or required status checks — treat it as an aid for human reviewers.
- Groq's free tier has request/token rate limits; the action retries with backoff and posts a partial review (flagging which chunks failed) rather than failing the whole job, unless `fail-on-review-error: true`.

## License

MIT — see [LICENSE](./LICENSE).
