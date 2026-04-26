

<!-- br-agent-instructions-v1 -->

---

## Beads Workflow Integration

**IMPORTANT: use beads to track your progress.**

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking. Issues are stored in `.beads/` and tracked in git.

**Note:** `br` is non-invasive and never executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Essential Commands

```bash
# View ready issues (unblocked, not deferred)
br ready

# List and search
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br search "keyword"   # Full-text search

# Create and update
br create --title="..." --description="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once

# Sync with git
br sync --flush-only  # Export DB to JSONL
br sync --status      # Check sync status
```

### Workflow Pattern

1. **Find work**: Run `br ready` to find actionable work.
2. **Take the bead**: Use `br update <id> --status=in_progress`.
3. **Implement**: Write the code for that bead.
4. **Test**: Add appropriate tests and run them. Prefer headless tests for renderer/GPU work where possible.
5. **Review**: Run the multi-perspective review.
6. **Address feedback**: Fix review findings. If review uncovers follow-up work, file beads for it.
7. **Complete**: Use `br close <id> --reason="Completed"` when the work is done.
8. **Sync**: Run `br sync --flush-only`, then stage `.beads/`.
9. **Commit**: Commit code, tests, docs, and beads changes together.
10. **Continue**: Proceed to the next ready bead until the overall task is done.

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4, not words)
- **Types**: task, bug, feature, epic, chore, docs, question
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads changes to JSONL
git add .beads/         # Stage beads JSONL updates
git commit -m "..."     # Commit everything
git push                # Only when explicitly requested
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress → closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always sync before ending session

<!-- end-br-agent-instructions -->
