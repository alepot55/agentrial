# Benchmark: ReAct Agent Evaluation

Real benchmark data from running a tool-calling agent through agentrial.

## Agent

A ReAct-style agent (`agent.py`) with 4 deterministic tools:
- **calculator** — evaluates math expressions
- **search_knowledge** — looks up factual knowledge from a hardcoded database
- **unit_converter** — converts between temperature, length, weight, and volume units
- **date_info** — answers date-related questions (days in year, leap year checks)

The agent uses raw Anthropic API calls (no frameworks) with Claude 3 Haiku.

## Results

**Model**: `claude-3-haiku-20240307` | **Trials**: 20 per case | **Total cost**: $0.11

| Test Case | Pass Rate | 95% CI | Avg Cost | Avg Latency |
|---|---|---|---|---|
| simple-math | 70% | (48.1% - 85.5%) | $0.0005 | 4734ms |
| capital-lookup | 100% | (83.9% - 100.0%) | $0.0005 | 3114ms |
| multi-step-calculation | 95% | (76.4% - 99.1%) | $0.0013 | 9618ms |
| unit-conversion | 100% | (83.9% - 100.0%) | $0.0011 | 8847ms |
| compound-reasoning | 95% | (76.4% - 99.1%) | $0.0011 | 6681ms |
| ambiguous-query | 100% | (83.9% - 100.0%) | $0.0009 | 7463ms |
| unnecessary-tool-call | 100% | (83.9% - 100.0%) | $0.0002 | 3349ms |

**Overall**: 94.3% (89.1% - 97.1%) | **ARS**: 79.7/100

## Key findings

1. **simple-math at 70%** — The model correctly calculates 247*18=4446 every time, but formats the output as "4,446" (with comma) ~30% of the time. The test expects "4446" without comma. This is a real-world formatting variance issue.

2. **multi-step-calculation at 95%** — Occasional rounding differences in multi-step tax calculations.

3. **unnecessary-tool-call at 100%** — The model correctly answers "blue" without calling any tools for "What color is the sky?", keeping cost at $0.0002 (lowest of all cases).

4. **Wilson CI width** — With 20 trials, a 100% pass rate gives CI of (83.9% - 100.0%). You need more trials to narrow the interval.

## Reproduce

```bash
pip install agentrial anthropic
export ANTHROPIC_API_KEY="your-key"
export AGENT_PROVIDER=anthropic
export AGENT_MODEL=claude-3-haiku-20240307

agentrial run examples/benchmark/agentrial.yml --trials 20 -o results.json
agentrial ars results.json
```
