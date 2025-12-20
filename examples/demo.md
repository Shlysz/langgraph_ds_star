# LangGraph DS-STAR Demo

This demo uses a small synthetic sales CSV to show the full pipeline: analyze -> plan -> code -> execute -> verify -> finalize.

## 1) Prepare

- Ensure you have an API key in `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`.
- The demo data file is `data/sample_sales.csv`.
- The demo config is `example_config.yaml` (live step output enabled, no artifact files by default).

## 2) Run

From the repo root:

```bash
python -m langgraph_dsstar.langgraph_dsstar --config example_config.yaml
```

## 2b) Invoke directly (no YAML)

You can run the graph directly in code and include absolute file paths in the query:

```python
from langgraph_dsstar.langgraph_dsstar import LGConfig, LangGraphDSStar

config = LGConfig(preserve_artifacts=False, emit_steps=True)
agent = LangGraphDSStar(config)
app = agent.build_graph()

state = {
    "query": "Please analyze /abs/path/to/data/sample_sales.csv and report results",
}
final_state = app.invoke(state)
print(final_state.get(\"final_result\", \"\"))
```

## 3) What to expect

By default, steps are printed live to the console (prompt, code, and result for each phase), and no intermediate artifacts are written.

The rough flow you should see in the console looks like this:

1. analyzer: summarizes `data/sample_sales.csv`
2. planner: proposes step(s) to answer the query
3. coder: generates analysis code
4. executor: runs the code and captures output
5. verifier: decides if the result answers the question
6. router: adds/removes plan steps if needed (optional)
7. finalyzer: generates final JSON answer code
8. final_exec: executes final code and saves `result.json`

## 4) Inspect output

```bash
cat runs/<run_id>/final_output/result.json
```

If you want to persist artifacts, set `preserve_artifacts: true` in `example_config.yaml`.
