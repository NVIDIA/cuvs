# Evaluation Report

Evaluation of the `accelerated-computing-cuvs` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `accelerated-computing-cuvs`
- Evaluation date: 2026-05-29
- NVSkills-Eval profile: `external`
- Environment: `local`
- Dataset: 15 evaluation tasks
- Attempts per task: 2
- Pass threshold: 50%
- Overall verdict: FAIL

## Agents Used

- `claude-code`
- `codex`

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- `security` (Security): checks for unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): verifies that the agent loaded the expected skill and workflow.
- `skill_efficiency` (Efficiency): checks routing quality, decoy avoidance, and redundant tool usage.
- `accuracy` (Accuracy): grades final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): checks whether the overall user task completed successfully.
- `behavior_check` (Behavior Check): verifies expected behavior steps, including safety expectations.
- `token_efficiency` (Token Efficiency): compares token usage with and without the skill.

## Test Tasks

The benchmark dataset contained 15 evaluation tasks:

- Positive tasks: 14 tasks where the skill was expected to activate.
- Negative tasks: 1 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 70% (-30%) | 77% (-13%) |
| Correctness | 8 | 94% (+5%) | 85% (+12%) |
| Discoverability | 8 | 80% (+16%) | 67% (+13%) |
| Effectiveness | 8 | 88% (+3%) | 78% (+6%) |
| Efficiency | 8 | 58% (+17%) | 48% (+8%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 11 total findings.

Top findings:

- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/accelerated-computing-cuvs/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/accelerated-computing-cuvs/SKILL.md`)
- LOW QUALITY/quality_discoverability: Description very long (229 chars, recommend 50-150) (`skills/accelerated-computing-cuvs/SKILL.md`)
- LOW QUALITY/quality_discoverability: Broad description without negative triggers may cause over-triggering (`skills/accelerated-computing-cuvs/SKILL.md`)
- LOW QUALITY/quality_discoverability: No '## Purpose' section (`skills/accelerated-computing-cuvs/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 3 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found across references/ann-patterns.md and references/api-index.md:
  "## Filtering Search Results" in references/ann-patterns.md (lines 222-247)
  vs "### Filtering" in references/api-index.md (lines 178-198) (`references/ann-patterns.md:222`)
- HIGH DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/api-index.md:
  "## Dtype Guidance" in SKILL.md (lines 136-153)
  vs "## Input Dtype Quick Map" in references/api-index.md (lines 73-92) (`SKILL.md:136`)
- HIGH DUPLICATE/duplicate: Duplicate content found within references/ann-patterns.md:
  "# 1. Prepare embeddings. float32 is the safe dense default; preserve a supported" in references/ann-patterns.md (lines 9-9)
  vs "# lower-precision dtype only when the chosen cuVS API and recall target support it." in references/ann-patterns.md (lines 10-13) (`references/ann-patterns.md:9`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
