# Rio Skills for OpenClaw — Implementation Plan
## Date: March 3, 2026

---

## Goal

Add two openclaw skills for Rio-Agent:
1. **rio-customer-care** — Customer care agent skill (handles support queries, ticket workflows, escalation, empathetic responses)
2. **rio-tutor** — Adaptive tutoring skill (detects student struggles, Socratic method, homework help, progress tracking, quiz generation)

These skills follow the openclaw `SKILL.md` manifest format and leverage Rio's existing capabilities:
- Voice conversation (Live API)
- Screen vision (on-demand + autonomous)
- Tool execution (read_file, write_file, patch_file, run_command)
- Struggle detection (4 screen-based signals)
- RAG memory (ChromaDB cross-session recall)

---

## Architecture Decisions

### Skills are OpenClaw SKILL.md packages
- Each skill lives in `openclaw/skills/rio-<name>/`
- SKILL.md frontmatter: name, description, metadata (emoji, requires)
- Body: instructions, workflows, command patterns
- Optional: `references/` for domain knowledge, `scripts/` for helper tools

### Rio integration via new Gemini tool declarations
- New tools added to `cloud/gemini_session.py`: `create_ticket`, `track_progress`, `generate_quiz`, `explain_concept`
- New tool handlers in `local/tools.py`
- System instruction extended for customer-care and tutor modes

### Skill activation
- Skills are context-injected when triggered by description match
- Rio's struggle detector naturally feeds into the tutor skill
- Customer care skill activates on support-related queries

---

## File Plan

### OpenClaw Skills (new files)
```
openclaw/skills/
├── rio-customer-care/
│   ├── SKILL.md                          # Manifest + instructions
│   └── references/
│       ├── escalation-workflows.md       # Escalation tiers & SLA rules
│       └── response-templates.md         # Empathetic response patterns
├── rio-tutor/
│   ├── SKILL.md                          # Manifest + instructions
│   ├── scripts/
│   │   └── quiz_generator.py             # Generate quizzes from topics
│   └── references/
│       ├── socratic-method.md            # Teaching methodology guide
│       └── learning-patterns.md          # Common student struggle patterns
```

### Rio-Agent modifications (existing files)
```
rio/cloud/gemini_session.py   — Add new tool declarations (customer-care + tutor tools)
rio/local/tools.py            — Add new tool handlers
rio/config.yaml               — Add skills section
rio/local/config.py           — Add SkillsConfig dataclass
```

---

## Task Checklist

- [ ] 1. Create `openclaw/skills/rio-customer-care/SKILL.md`
- [ ] 2. Create `openclaw/skills/rio-customer-care/references/escalation-workflows.md`
- [ ] 3. Create `openclaw/skills/rio-customer-care/references/response-templates.md`
- [ ] 4. Create `openclaw/skills/rio-tutor/SKILL.md`
- [ ] 5. Create `openclaw/skills/rio-tutor/references/socratic-method.md`
- [ ] 6. Create `openclaw/skills/rio-tutor/references/learning-patterns.md`
- [ ] 7. Create `openclaw/skills/rio-tutor/scripts/quiz_generator.py`
- [ ] 8. Add customer-care + tutor tool declarations to `rio/cloud/gemini_session.py`
- [ ] 9. Add customer-care + tutor tool handlers to `rio/local/tools.py`
- [ ] 10. Add skills config to `rio/config.yaml` and `rio/local/config.py`
- [ ] 11. Update `rio/context.txt` with skills documentation
- [ ] 12. Update `rio/tasks/todo.md` with skills completion status
- [ ] 13. Verify all files parse correctly

---

## Verification Criteria

1. All SKILL.md files have valid YAML frontmatter
2. All scripts are executable and have proper error handling
3. New tool declarations follow existing Rio patterns (dict-based Schema)
4. New tool handlers follow graceful degradation pattern
5. Config changes are backward-compatible (new sections, no breakage)
6. Context.txt accurately documents new capabilities
