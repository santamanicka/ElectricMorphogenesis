# Project Migration Workflow: → Box + GitHub

This documents how to migrate a project to Box cloud storage and GitHub, preserving full git history. Covers three scenarios:

- **Scenario A**: Project exists locally (e.g., in `~/PycharmProjects/`)
- **Scenario B**: Project only exists on a remote (e.g., GitLab) — no local copy
- **Scenario C**: Push a local or remote repo into an *existing* GitHub repo (same branch or new branch)

## CAUTION

- **Always COPY, never MOVE.** Use `rsync`, `cp`, or `git clone` — never `mv`. The original source (local project or remote repo) must remain intact.
- **Never delete existing files.** Do not remove the original project after copying. Do not delete files from an existing GitHub repo before pushing. If something goes wrong, the original is your safety net.
- When using `--force` on a push, understand that it overwrites the *remote* branch history — make sure the remote has nothing you need that isn't also in your local copy.

## Prerequisites

- GitHub CLI (`gh`) installed and authenticated
- Box cloud storage mounted at `~/Library/CloudStorage/Box-Box/`
- Git installed

---

# Scenario A: Local Project → Box + GitHub

Use this when the project already exists on your machine (e.g., `~/PycharmProjects/<project>`).

## Step A1: Create a GitHub repo

Create an **empty** repo on GitHub (no README, no .gitignore, no license). This avoids conflicts when pushing existing history.

```bash
gh repo create <github-username>/<repo-name> --public
# or --private for private repos
```

If you already created a repo with an initial commit, that's fine — we'll force-push over it in Step A4.

## Step A2: Copy project to Box (with .git history)

Use `rsync` to copy the project, **including the `.git/` directory**, while excluding virtual environments and build artifacts:

```bash
rsync -a \
  --exclude='.venv/' \
  --exclude='grn-env/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='*.egg-info/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='.DS_Store' \
  ~/PycharmProjects/<project-name>/ \
  ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<project-name>/
```

**Key**: Do NOT exclude `.git/` — this is what preserves the commit history.

Verify the history copied correctly:

```bash
git -C ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<project-name> log --oneline | head -5
```

## Step A3: Update .gitignore

Update `.gitignore` in the Box copy to exclude Claude-related files and any other files you don't want tracked:

```
# Claude
.claude_conversations/
.claude/
CLAUDE.md
MEMORY.md

# Python
python_cache/
__pycache__/
*.pyc
*.egg-info/

# Virtual environments
.venv/

# Build
build/
dist/

# IDE
.idea/
.vscode/

# OS
.DS_Store
```

Adjust the patterns based on your project's needs.

## Step A4: Update remote and push to GitHub

The copied `.git` directory still points to the old remote (e.g., GitLab). Update it to GitHub:

```bash
cd ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<project-name>

# Change remote from old origin (e.g., GitLab) to GitHub
git remote set-url origin https://github.com/<github-username>/<repo-name>.git

# Verify
git remote -v
```

Push the full history. If the GitHub repo already has commits (like an "Initial commit"), use `--force`:

```bash
# If GitHub repo is empty:
git push -u origin main

# If GitHub repo already has an initial commit:
git push --force origin main
```

## Step A5: Verify

```bash
# Check latest commits on GitHub
gh api repos/<github-username>/<repo-name>/commits --jq '.[].commit.message' | head -5

# Check commit count
git log --oneline | wc -l
```

---

# Scenario B: Remote-Only Repo (e.g., GitLab) → Box + GitHub

Use this when the project exists only on a remote like GitLab and you don't have a local copy.

## Step B1: Create a GitHub repo

Same as Scenario A — create an **empty** repo:

```bash
gh repo create <github-username>/<repo-name> --public
# or --private for private repos
```

## Step B2: Clone from the remote directly into Box

Clone the full repo (with all history) straight into the Box folder. Use `--bare` first for a clean transfer, then convert, OR clone normally:

**Option 1: Normal clone (simpler)**

```bash
git clone https://gitlab.com/<gitlab-username>/<repo-name>.git \
  ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<repo-name>
```

**Option 2: Mirror clone (preserves all branches and tags)**

```bash
# Clone as a bare mirror
git clone --mirror https://gitlab.com/<gitlab-username>/<repo-name>.git \
  ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<repo-name>/.git

# Convert from bare to a normal working repo
cd ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<repo-name>
git config --bool core.bare false
git checkout main
```

The mirror option is better if the remote has multiple branches or tags you want to preserve.

Verify the history:

```bash
git -C ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<repo-name> log --oneline | head -5
```

## Step B3: Update .gitignore

Same as Step A3 — add Claude and other exclusions to `.gitignore`.

## Step B4: Update remote to GitHub and push

The origin still points to GitLab. Switch it to GitHub:

```bash
cd ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<repo-name>

# Replace GitLab remote with GitHub
git remote set-url origin https://github.com/<github-username>/<repo-name>.git

# Verify
git remote -v

# Push all branches and tags
git push -u origin main
git push origin --tags

# If you used --mirror and want to push ALL branches:
git push --all origin
git push --tags origin
```

If the GitHub repo already has an initial commit:

```bash
git push --force origin main
```

## Step B5: (Optional) Keep GitLab as a secondary remote

If you want to keep a reference to the original GitLab repo:

```bash
git remote add gitlab https://gitlab.com/<gitlab-username>/<repo-name>.git
git remote -v
# origin   https://github.com/...  (GitHub - primary)
# gitlab   https://gitlab.com/...  (GitLab - reference)
```

## Step B6: Verify

Same as Step A5:

```bash
gh api repos/<github-username>/<repo-name>/commits --jq '.[].commit.message' | head -5
git log --oneline | wc -l
```

---

# Scenario C: Push into an Existing GitHub Repo

Use this when a GitHub repo already exists and you want to add code from a local project or a remote (e.g., GitLab) — either into the existing branch (e.g., `main`) or into a separate branch.

## Step C1: Get the source repo locally

**If the source is local** (e.g., `~/PycharmProjects/<project>`), just `cd` into it:

```bash
cd ~/PycharmProjects/<project-name>
```

**If the source is remote-only** (e.g., GitLab), clone it to a temporary location:

```bash
git clone https://gitlab.com/<gitlab-username>/<repo-name>.git /tmp/<repo-name>
cd /tmp/<repo-name>
```

## Step C2: Add the existing GitHub repo as a remote

```bash
# If "origin" already points somewhere else, add GitHub as a named remote
git remote add github https://github.com/<github-username>/<existing-repo>.git

# Fetch the GitHub repo's history so git knows about its branches
git fetch github
```

## Step C3: Choose your strategy

### Option 1: Replace the existing branch

This overwrites the GitHub branch's history with the source repo's history.

```bash
# Force-push to overwrite the existing branch
git push --force github main
```

**Warning**: This rewrites history on GitHub. Use only if you're replacing a placeholder (like a bare "Initial commit") or you're certain you want to overwrite.

### Option 2: Push into a new branch

This keeps the existing GitHub content untouched and adds your source repo's history as a separate branch.

```bash
# Push the source repo's main branch as a new branch on GitHub
git push github main:<new-branch-name>

# Examples:
git push github main:from-gitlab
git push github main:legacy-codebase
git push github main:v1-archive
```

Verify the branch was created:

```bash
gh api repos/<github-username>/<existing-repo>/branches --jq '.[].name'
```

### Option 3: Merge histories (combine both repos' commits)

This merges the source repo's history into the existing GitHub branch. Useful when both repos have meaningful history you want to keep.

```bash
# Fetch GitHub's history
git fetch github

# Allow merging unrelated histories (since the repos have different roots)
git merge github/main --allow-unrelated-histories -m "Merge existing GitHub history with migrated repo"

# Push the combined history
git push github main
```

**Note**: This may produce merge conflicts if both repos have files with the same names. Resolve them before pushing.

## Step C4: (Optional) Copy to Box

If you also want a Box copy:

```bash
rsync -a \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='*.egg-info/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='.DS_Store' \
  /path/to/source-repo/ \
  ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<project-name>/

# Update the Box copy's remote to point to GitHub
git -C ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<project-name> \
  remote set-url origin https://github.com/<github-username>/<existing-repo>.git
```

## Step C5: Clean up

If you cloned to `/tmp` in Step C1:

```bash
rm -rf /tmp/<repo-name>
```

## Step C6: Verify

```bash
# Check branches on GitHub
gh api repos/<github-username>/<existing-repo>/branches --jq '.[].name'

# Check latest commits (on main or your new branch)
gh api repos/<github-username>/<existing-repo>/commits?sha=<branch-name> \
  --jq '.[].commit.message' | head -5
```

---

# Common Final Steps (All Scenarios)

## (Optional) Set up .claude_conversations

Organize Claude conversation logs and config files into a `.claude_conversations/` folder inside the Box project. This keeps them backed up in Box and accessible when working from the Box copy.

### How Claude Code stores conversations

Claude Code stores conversation logs under `~/.claude/projects/`, using an encoded version of the project's absolute path as the folder name. For example:

- Local project at `~/PycharmProjects/grnmemory` → conversations stored in:
  `~/.claude/projects/-Users-santoshmanicka-PycharmProjects-grnmemory/`
- Box project at `~/Library/CloudStorage/Box-Box/My Tufts/Research/Code/grnmemory` → conversations expected at:
  `~/.claude/projects/-Users-santoshmanicka-Library-CloudStorage-Box-Box-My-Tufts-Research-Code-grnmemory`

Each folder contains `.jsonl` files (one per conversation session) and subfolders for session data.

### Step 1: Create `.claude_conversations/` in the Box project

```bash
cd ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<project-name>
mkdir -p .claude_conversations
```

### Step 2: Copy Claude-related files into it

Copy (not move!) any Claude config and markdown files from the Box project root:

```bash
cp CLAUDE.md .claude_conversations/
cp -R .claude/ .claude_conversations/
# Copy any other conversation-related markdown files, e.g.:
# cp "Memory in GRN.md" .claude_conversations/
```

### Step 3: Copy existing conversation logs (if migrating from a local repo)

If the project previously lived locally (e.g., in `~/PycharmProjects/`), there may be existing conversation logs stored under `~/.claude/projects/`. Copy them into `.claude_conversations/`:

```bash
# Find the encoded project path under ~/.claude/projects/
# Replace dashes and slashes with hyphens: ~/PycharmProjects/<name> becomes:
# -Users-<username>-PycharmProjects-<project-name>

cp -R ~/.claude/projects/-Users-<username>-PycharmProjects-<project-name>/* \
  ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<project-name>/.claude_conversations/
```

Example for grnmemory:

```bash
cp -R ~/.claude/projects/-Users-santoshmanicka-PycharmProjects-grnmemory/* \
  ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/grnmemory/.claude_conversations/
```

### Step 4: Create a symlink so Claude Code finds conversations for the Box path

When you open the project from the Box location, Claude Code looks for conversations at the encoded Box path under `~/.claude/projects/`. Create a symlink pointing that path to `.claude_conversations/`:

```bash
ln -s ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/<project-name>/.claude_conversations \
  ~/.claude/projects/-Users-<username>-Library-CloudStorage-Box-Box-My-Tufts-Research-Code-<project-name>
```

Example for grnmemory:

```bash
ln -s /Users/santoshmanicka/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/grnmemory/.claude_conversations \
  ~/.claude/projects/-Users-santoshmanicka-Library-CloudStorage-Box-Box-My-Tufts-Research-Code-grnmemory
```

### Verify the symlink

```bash
ls -la ~/.claude/projects/ | grep <project-name>
# Should show: ... -> /Users/.../Box-Box/.../project-name/.claude_conversations
```

### Result

After this setup:
- Conversation logs live inside the Box project at `.claude_conversations/`
- They sync to Box cloud automatically
- Claude Code finds them via the symlink when you open the project from Box
- They are excluded from git via `.gitignore` (from Step A3/B3)

## Quick Reference

| Step | Scenario A (local) | Scenario B (remote-only) | Scenario C (into existing repo) |
|------|-------------------|-------------------------|-------------------------------|
| 1 | `gh repo create` | `gh repo create` | Get source locally (`cd` or `git clone`) |
| 2 | `rsync -a` (with `.git/`) | `git clone` or `git clone --mirror` | `git remote add github` + `git fetch` |
| 3 | Edit `.gitignore` | Edit `.gitignore` | Choose: replace, new branch, or merge |
| 4 | `git remote set-url` + `git push` | `git remote set-url` + `git push` | `git push github ...` |
| 5 | Verify | Verify | Verify |

### Scenario C Strategy Summary

| Goal | Command |
|------|---------|
| Replace existing branch | `git push --force github main` |
| Add as new branch | `git push github main:<new-branch>` |
| Merge both histories | `git merge github/main --allow-unrelated-histories` then `git push` |

## Example: grnmemory (Scenario A)

```bash
# Copied 146 commits from GitLab origin to GitHub
rsync -a --exclude='.venv/' --exclude='grn-env/' --exclude='__pycache__/' \
  ~/PycharmProjects/grnmemory/ \
  ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/grnmemory/

git -C ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/grnmemory \
  remote set-url origin https://github.com/santamanicka/grnmemory.git

git -C ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/grnmemory \
  push --force origin main
```

## Example: Scenario B (hypothetical)

```bash
# Clone a GitLab-only project directly into Box
git clone --mirror https://gitlab.com/smanicka/some-project.git \
  ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/some-project/.git

cd ~/Library/CloudStorage/Box-Box/"My tufts"/Research/Code/some-project
git config --bool core.bare false
git checkout main

# Point to GitHub and push
git remote set-url origin https://github.com/santamanicka/some-project.git
git push --all origin
git push --tags origin
```

## Example: Scenario C (push GitLab repo into existing GitHub repo as new branch)

```bash
# Clone the GitLab source to a temp location
git clone https://gitlab.com/smanicka/old-project.git /tmp/old-project
cd /tmp/old-project

# Add the existing GitHub repo as a remote
git remote add github https://github.com/santamanicka/existing-repo.git
git fetch github

# Push as a new branch called "from-gitlab"
git push github main:from-gitlab

# Clean up
rm -rf /tmp/old-project
```