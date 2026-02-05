# LangGraph + Claude Haiku Example

This example demonstrates agentrial with a real LangGraph agent using Claude 3 Haiku.

## Setup

1. Install dependencies:

```bash
pip install agentrial[langgraph] langchain-anthropic
```

2. Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Run

```bash
cd examples/langgraph_haiku
agentrial run
```

## Expected Output

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ langgraph-haiku-demo - PASSED                                                │
╰────────────────────────────────────────────────────── Threshold: 85.0% ──────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Test Case                 ┃ Pass Rate┃ 95% CI          ┃ Avg Cost ┃ Avg Lat. ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ easy-multiply             │   100.0% │ (72.2%-100.0%)  │  $0.0005 │    1.59s │
│ easy-capital-japan        │   100.0% │ (72.2%-100.0%)  │  $0.0004 │    1.60s │
│ easy-celsius-to-fahrenheit│   100.0% │ (72.2%-100.0%)  │  $0.0005 │    1.90s │
│ medium-population-france  │   100.0% │ (72.2%-100.0%)  │  $0.0005 │    1.55s │
│ medium-power-of-two       │   100.0% │ (72.2%-100.0%)  │  $0.0005 │    1.57s │
│ hard-multi-step           │   100.0% │ (72.2%-100.0%)  │  $0.0011 │    4.03s │
│ edge-unknown-country      │   100.0% │ (72.2%-100.0%)  │  $0.0003 │    1.26s │
└───────────────────────────┴──────────┴─────────────────┴──────────┴──────────┘

Overall Pass Rate: 100.0% (96.3%-100.0%)
Total Cost: $0.04
```

## Cost Breakdown

With 10 trials per test case (70 total trials):

| Test Type | Cost/Trial | Tokens |
|-----------|-----------|--------|
| Easy      | ~$0.0005  | ~1,500 |
| Medium    | ~$0.0005  | ~1,500 |
| Hard      | ~$0.0011  | ~3,500 |

**Total: ~$0.04 for full test suite**

## Files

- `agent.py` - LangGraph agent with 3 tools (calculate, lookup_country_info, convert_temperature)
- `agentrial.yml` - Test cases covering easy, medium, hard, and edge cases
