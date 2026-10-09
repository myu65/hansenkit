# Continue in Codex

Open this repository's local directory as a Codex project (the Git root containing `AGENTS.md`).
Repository instructions are discovered from that root according to the
[official AGENTS.md documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
No global Codex settings or existing projects need to change. This bootstrap provisions a Git
repository and project instructions; adding the directory to the desktop sidebar is a UI operation.

Useful first prompt:

> Read AGENTS.md and docs/evaluation.md. Work on issue #2 by preparing a dataset-clearance
> checklist and the independent neutral-molecule benchmark protocol from issue #4. Keep all
> unapproved HSP tables and weights blocked. Use a focused branch and PR; do not download
> third-party numerical data or train on them until their permitted uses are documented.

The M0 PoC issue and its PR demonstrate Issue → PR → tests → CI → merge. M1–M3 work is tracked
in [GitHub milestones](https://github.com/myu65/hansenkit/milestones) and [issues](https://github.com/myu65/hansenkit/issues).
