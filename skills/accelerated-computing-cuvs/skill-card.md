## Description: <br>
Official NVIDIA-authored guidance for NVIDIA cuVS GPU vector search, approximate nearest neighbors, CAGRA, IVF-Flat, IVF-PQ, HNSW export, embedding retrieval, semantic search, vector database backends, RAG search, GPU clustering. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building GPU-accelerated vector search pipelines or integrating cuVS into database and search platform backends for approximate nearest neighbor search, embedding retrieval, and RAG workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuVS Documentation](https://docs.nvidia.com/cuvs/home) <br>
- [Integration Patterns](https://docs.nvidia.com/cuvs/user-guide/integration-patterns) <br>
- [Compatibility](https://docs.nvidia.com/cuvs/user-guide/compatibility) <br>
- [cuVS GitHub](https://github.com/rapidsai/cuvs) <br>
- [cuVS CHANGELOG](https://github.com/rapidsai/cuvs/blob/main/CHANGELOG.md) <br>
- [Index Selection Guide](references/index-selection.md) <br>
- [ANN Patterns](references/ann-patterns.md) <br>
- [API Index](references/api-index.md) <br>
- [Version Boundaries](references/version-boundaries.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 15 internal evaluation tasks (14 positive skill-activation, 1 negative activation) with 2 attempts per task using NVSkills-Eval external profile. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 70% (-30%) | 77% (-13%) |
| Correctness | 8 | 94% (+5%) | 85% (+12%) |
| Discoverability | 8 | 80% (+16%) | 67% (+13%) |
| Effectiveness | 8 | 88% (+3%) | 78% (+6%) |
| Efficiency | 8 | 58% (+17%) | 48% (+8%) |

## Skill Version(s): <br>
ac5e621 (source: git SHA, committed 2026-05-29) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
