import os
import json
import subprocess
import re
import uuid
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict

import yaml
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langgraph_dsstar.prompts import (
    analyzer,
    planner_init,
    planner_next,
    coder_init,
    coder_next,
    verifier,
    router,
    debugger,
    finalyzer,
)


@dataclass
class LGConfig:
    run_id: Optional[str] = None
    max_refinement_rounds: int = 5
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    interactive: bool = False
    auto_debug: bool = True
    debug_attempts: float = float("inf")
    execution_timeout: int = 60
    preserve_artifacts: bool = True
    runs_dir: str = "runs"
    data_dir: str = "data"
    agent_models: Dict[str, str] = field(default_factory=dict)
    emit_steps: bool = True
    display_char_limit: int = 1200

    def __post_init__(self) -> None:
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        if self.agent_models is None:
            self.agent_models = {}


class ArtifactStorage:
    """Save every step artifact for reproducibility."""

    def __init__(self, config: LGConfig):
        self.config = config
        self.run_dir = Path(config.runs_dir) / config.run_id
        if self.config.preserve_artifacts:
            self._setup_directories()

    def _setup_directories(self) -> None:
        dirs = [
            self.run_dir,
            self.run_dir / "steps",
            self.run_dir / "data_cache",
            self.run_dir / "logs",
            self.run_dir / "final_output",
            self.run_dir / "exec_env",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def save_step(
        self,
        step_id: str,
        step_type: str,
        prompt: str,
        code: Optional[str],
        result: str,
        metadata: Dict[str, Any],
    ) -> None:
        step_dir = self.run_dir / "steps" / step_id
        step_dir.mkdir(exist_ok=True)

        (step_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        if code:
            (step_dir / "code.py").write_text(code, encoding="utf-8")
        (step_dir / "result.txt").write_text(result, encoding="utf-8")

        metadata.update(
            {
                "timestamp": datetime.now().isoformat(),
                "step_type": step_type,
                "step_id": step_id,
            }
        )
        (step_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def list_steps(self) -> List[Dict[str, Any]]:
        if not self.config.preserve_artifacts:
            return []
        steps = []
        for step_path in sorted(self.run_dir.glob("steps/*")):
            meta_path = step_path / "metadata.json"
            if meta_path.exists():
                steps.append(json.loads(meta_path.read_text()))
        return steps


class GraphState(TypedDict, total=False):
    query: str
    messages: List[Any]
    data_files: List[str]
    absolute_data_files: List[str]
    data_descriptions: Dict[str, str]
    data_desc_str: str
    plan: List[str]
    code: Optional[str]
    exec_result: Optional[str]
    verdict: Optional[str]
    routing: Optional[str]
    round_idx: int
    max_refinement_rounds: int
    final_code: Optional[str]
    final_result: Optional[str]
    emit_steps: Optional[bool]
    preserve_artifacts: Optional[bool]
    display_char_limit: Optional[int]
    execution_timeout: Optional[int]


class LangGraphDSStar:
    """LangGraph-based implementation of DS-STAR pipeline."""

    def __init__(self, config: LGConfig) -> None:
        # Basic setup for run config, artifacts, and providers.
        self.config = config
        self.storage = ArtifactStorage(config) if self.config.preserve_artifacts else None
        self.model = self._init_model()
        self._step_counter = 0

    def _init_model(self) -> ChatOpenAI:
        # Single model for all agents, initialized in the requested format.
        api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            api_key=api_key,
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            temperature=0.7,
        )

    def _apply_runtime_overrides(self, state: GraphState) -> None:
        # Allow per-invoke overrides from state for frontend usage.
        if "emit_steps" in state:
            self.config.emit_steps = bool(state["emit_steps"])
        if "preserve_artifacts" in state:
            self.config.preserve_artifacts = bool(state["preserve_artifacts"])
        if "display_char_limit" in state and state["display_char_limit"] is not None:
            self.config.display_char_limit = int(state["display_char_limit"])
        if "execution_timeout" in state and state["execution_timeout"] is not None:
            self.config.execution_timeout = int(state["execution_timeout"])

    def _extract_paths_from_query(self, query: str) -> List[str]:
        # Extract absolute paths from query text.
        if not query:
            return []
        candidates = re.findall(r"(/[^\\s\"']+)", query)
        return list(dict.fromkeys(candidates))

    def _extract_query_from_messages(self, state: GraphState) -> str:
        messages = state.get("messages") or []
        if not messages:
            return ""
        last = messages[-1]
        content = getattr(last, "content", None)
        if content:
            return content
        return str(last)

    def _next_step_id(self, step_name: str) -> str:
        step_id = f"{self._step_counter:03d}_{step_name}"
        self._step_counter += 1
        return step_id

    def _truncate(self, text: str) -> str:
        if text is None:
            return ""
        if len(text) <= self.config.display_char_limit:
            return text
        return text[: self.config.display_char_limit] + "\n...[truncated]..."

    def _emit_step(
        self,
        step_type: str,
        prompt: str,
        code: Optional[str],
        result: str,
        metadata: Dict[str, Any],
    ) -> None:
        if not self.config.emit_steps:
            return
        step_id = metadata.get("step_id", "step")
        print(f"\n{'='*60}")
        print(f"STEP {step_id}: {step_type}")
        if prompt:
            print("\n[PROMPT]")
            print(self._truncate(prompt))
        if code:
            print("\n[CODE]")
            print(self._truncate(code))
        if result:
            print("\n[RESULT]")
            print(self._truncate(result))
        print(f"{'='*60}")

    def _save_step(
        self,
        step_type: str,
        prompt: str,
        code: Optional[str],
        result: str,
        metadata: Dict[str, Any],
    ) -> None:
        step_id = self._next_step_id(step_type)
        metadata = dict(metadata)
        metadata["step_id"] = step_id
        self._emit_step(step_type, prompt, code, result, metadata)
        if self.config.preserve_artifacts and self.storage:
            self.storage.save_step(step_id, step_type, prompt, code, result, metadata)
        if self.config.interactive:
            input(f"Step {step_id} complete. Press Enter to continue...")

    def _call_model(self, agent_name: str, prompt: str) -> str:
        # Use a single ChatOpenAI instance for all agents.
        response = self.model.invoke(prompt)
        return response.content

    def _extract_code_block(self, response: str) -> str:
        code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
        return code_blocks[0] if code_blocks else response.strip()

    def _execute_code(self, code_script: str, data_files: Optional[List[str]] = None) -> (str, Optional[str]):
        # Execute generated code in a separate process.
        if data_files:
            missing = []
            for f in data_files:
                p = Path(f)
                if p.is_absolute():
                    if not p.exists():
                        missing.append(f)
                else:
                    if not (Path(self.config.data_dir) / f).exists():
                        missing.append(f)
            if missing:
                return "", f"Missing data files: {missing}"

        exec_path = None
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        try:
            temp_file.write(code_script)
            temp_file.flush()
            exec_path = temp_file.name
        finally:
            temp_file.close()

        try:
            result = subprocess.run(
                [sys.executable, str(exec_path)],
                capture_output=True,
                text=True,
                timeout=self.config.execution_timeout,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                return result.stdout, None
            error_msg = result.stderr or "Unknown execution error"
            return "", error_msg
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {self.config.execution_timeout}s"
        except Exception as e:
            return "", f"Execution error: {str(e)}"
        finally:
            if exec_path:
                try:
                    Path(exec_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _execute_and_debug_code(self, code: str, data_files: List[str], data_desc: str) -> str:
        # Run code and auto-fix on error with the debugger agent.
        exec_result, error = self._execute_code(code, data_files)
        attempts = 0
        while error and self.config.auto_debug and attempts < self.config.debug_attempts:
            # 如果是不可恢复错误直接终止程序
            if "Missing data files" in error:
                break
            code = self._debug_code(code, error, data_desc, data_files)
            exec_result, error = self._execute_code(code, data_files)
            attempts += 1
        return exec_result

    def _debug_code(self, code: str, error: str, data_desc: str, filenames: List[str]) -> str:
        # Ask the debugger agent to fix the code based on the error message.
        prompt = debugger.PROMPT.format(
            summaries=data_desc,
            code=code,
            bug=error,
            filenames=", ".join(filenames),
        )
        response = self._call_model("DEBUGGER", prompt)
        fixed_code = self._extract_code_block(response)
        self._save_step(
            "debugger",
            prompt=prompt,
            code=fixed_code,
            result=error,
            metadata={"error_type": error.split(":")[0]},
        )
        return fixed_code

    def analyze_files(self, state: GraphState) -> GraphState:
        # Phase 1: analyze each file and collect summaries.
        self._apply_runtime_overrides(state)
        if self.config.preserve_artifacts and self.storage is None:
            self.storage = ArtifactStorage(self.config)
        if self.config.preserve_artifacts:
            Path(self.config.data_dir).mkdir(exist_ok=True)
        query = state.get("query") or self._extract_query_from_messages(state)
        data_descriptions: Dict[str, str] = {}
        absolute_data_files: List[str] = []
        data_files = state.get("data_files") or self._extract_paths_from_query(query)
        if not data_files:
            raise ValueError("No data files provided; pass data_files or include absolute paths in query.")
        for f in data_files:
            p = Path(f)
            abs_path = str(p.resolve()) if p.is_absolute() else str(Path(self.config.data_dir).joinpath(f).resolve())
            absolute_data_files.append(abs_path)
            prompt = analyzer.PROMPT.format(filename=abs_path)
            response = self._call_model("ANALYZER", prompt)
            code = self._extract_code_block(response)
            exec_result = self._execute_and_debug_code(code, [abs_path], data_desc="")
            data_descriptions[abs_path] = exec_result
            self._save_step(
                "analyzer",
                prompt=prompt,
                code=code,
                result=exec_result,
                metadata={"filename": abs_path},
            )
        data_desc_str = "\n".join([f"File: {k}\n{v}" for k, v in data_descriptions.items()])
        return {
            "data_descriptions": data_descriptions,
            "absolute_data_files": absolute_data_files,
            "data_desc_str": data_desc_str,
            "query": query,
        }

    def plan_next(self, state: GraphState) -> GraphState:
        # Generate the next plan step based on current summaries and results.
        plan = state.get("plan", [])
        if not plan:
            prompt = planner_init.PROMPT.format(question=state["query"], summaries=state["data_desc_str"])
            response = self._call_model("PLANNER", prompt)
            step = response.strip()
        else:
            plan_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan))
            prompt = planner_next.PROMPT.format(
                question=state["query"],
                summaries=state["data_desc_str"],
                plan=plan_str,
                result=state.get("exec_result", ""),
                current_step=plan[-1],
            )
            response = self._call_model("PLANNER", prompt)
            step = response.strip()
        plan.append(step)
        self._save_step(
            "planner",
            prompt=prompt,
            code=None,
            result=step,
            metadata={"plan_length": len(plan)},
        )
        return {"plan": plan}

    def generate_code(self, state: GraphState) -> GraphState:
        # Generate or extend code based on the plan.
        plan = state.get("plan", [])
        plan_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan))
        if not state.get("code"):
            prompt = coder_init.PROMPT.format(summaries=state["data_desc_str"], plan=plan_str)
        else:
            prompt = coder_next.PROMPT.format(
                summaries=state["data_desc_str"],
                base_code=state["code"],
                plan=plan_str,
                current_plan=plan[-1],
            )
        response = self._call_model("CODER", prompt)
        code = self._extract_code_block(response)
        self._save_step(
            "coder",
            prompt=prompt,
            code=code,
            result="generated",
            metadata={"plan_length": len(plan)},
        )
        return {"code": code}

    def execute_code(self, state: GraphState) -> GraphState:
        # Execute the current code and store the result.
        exec_result = self._execute_and_debug_code(
            state["code"], state["absolute_data_files"], state["data_desc_str"]
        )
        self._save_step(
            "executor",
            prompt="",
            code=state["code"],
            result=exec_result,
            metadata={"phase": "analysis"},
        )
        return {"exec_result": exec_result}

    def verify(self, state: GraphState) -> GraphState:
        # Verify if the current result is sufficient to answer the query.
        plan = state.get("plan", [])
        plan_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan))
        prompt = verifier.PROMPT.format(
            plan=plan_str,
            code=state["code"],
            result=state["exec_result"],
            question=state["query"],
            summaries=state["data_desc_str"],
            current_step=plan[-1],
        )
        response = self._call_model("VERIFIER", prompt)
        verdict = response.strip()
        self._save_step(
            "verifier",
            prompt=prompt,
            code=None,
            result=verdict,
            metadata={"plan_length": len(plan)},
        )
        return {"verdict": verdict}

    def route_plan(self, state: GraphState) -> GraphState:
        # Decide whether to add a step or remove a wrong step.
        plan = state.get("plan", [])
        plan_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan))
        prompt = router.PROMPT.format(
            question=state["query"],
            summaries=state["data_desc_str"],
            plan=plan_str,
            result=state["exec_result"],
            current_step=plan[-1],
        )
        response = self._call_model("ROUTER", prompt).strip()
        routing = response

        lowered = routing.lower()
        if lowered.startswith("step"):
            parts = routing.split()
            if len(parts) >= 2 and parts[1].isdigit():
                step_to_remove = int(parts[1]) - 1
                if step_to_remove >= 0:
                    plan = plan[:step_to_remove]

        self._save_step(
            "router",
            prompt=prompt,
            code=None,
            result=routing,
            metadata={"plan_length": len(plan)},
        )
        return {"plan": plan, "routing": routing, "round_idx": state.get("round_idx", 0) + 1}

    def finalize(self, state: GraphState) -> GraphState:
        # Generate final answer code with required output format.
        prompt = finalyzer.PROMPT.format(
            summaries=state["data_desc_str"],
            code=state["code"],
            result=state["exec_result"],
            question=state["query"],
            guidelines="请输出 JSON，键名为 final_answer",
        )
        response = self._call_model("FINALYZER", prompt)
        final_code = self._extract_code_block(response)
        self._save_step(
            "finalyzer",
            prompt=prompt,
            code=final_code,
            result="generated",
            metadata={},
        )
        return {"final_code": final_code}

    def execute_final(self, state: GraphState) -> GraphState:
        # Execute the final answer code and persist output.
        final_result = self._execute_and_debug_code(
            state["final_code"], state["absolute_data_files"], state["data_desc_str"]
        )
        output_file = None
        if self.config.preserve_artifacts and self.storage:
            output_file = self.storage.run_dir / "final_output" / "result.json"
            output_file.write_text(final_result)
        self._save_step(
            "final_exec",
            prompt="",
            code=state["final_code"],
            result=final_result,
            metadata={"output_file": str(output_file) if output_file else ""},
        )
        return {"final_result": final_result}

    def _decide_next(self, state: GraphState) -> str:
        # Decide whether to refine or finalize based on verdict and round count.
        verdict = (state.get("verdict") or "").strip().lower()
        if verdict == "yes":
            return "finalize"
        if state.get("round_idx", 0) >= state.get("max_refinement_rounds", 0):
            return "finalize"
        return "refine"

    def build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("analyze_files", self.analyze_files)
        graph.add_node("plan_next", self.plan_next)
        graph.add_node("generate_code", self.generate_code)
        graph.add_node("execute_code", self.execute_code)
        graph.add_node("verify", self.verify)
        graph.add_node("route_plan", self.route_plan)
        graph.add_node("finalize", self.finalize)
        graph.add_node("execute_final", self.execute_final)

        graph.set_entry_point("analyze_files")
        graph.add_edge("analyze_files", "plan_next")
        graph.add_edge("plan_next", "generate_code")
        graph.add_edge("generate_code", "execute_code")
        graph.add_edge("execute_code", "verify")

        graph.add_conditional_edges(
            "verify",
            self._decide_next,
            {
                "refine": "route_plan",
                "finalize": "finalize",
            },
        )

        graph.add_edge("route_plan", "plan_next")
        graph.add_edge("finalize", "execute_final")
        graph.add_edge("execute_final", END)

        return graph.compile()

    def run(self, query: str, data_files: Optional[List[str]] = None) -> Dict[str, Any]:
        # Run the full pipeline graph with the initial state.
        app = self.build_graph()
        initial_state: GraphState = {
            "query": query,
            "plan": [],
            "code": None,
            "exec_result": None,
            "round_idx": 0,
            "max_refinement_rounds": self.config.max_refinement_rounds,
        }
        if data_files:
            initial_state["data_files"] = data_files
        final_state = app.invoke(initial_state)

        return {
            "run_id": self.config.run_id,
            "final_result": final_state.get("final_result", ""),
            "output_file": str(self.storage.run_dir / "final_output" / "result.json")
            if (self.config.preserve_artifacts and self.storage)
            else "",
            "total_steps": self._step_counter,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph DS-STAR")
    parser.add_argument("--data-files", nargs="+", help="Data files to analyze")
    parser.add_argument("--query", type=str, help="Analysis query")
    parser.add_argument("--max-rounds", type=int, help="Max refinement rounds")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--interactive", action="store_true", help="Pause between steps")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config_defaults = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config_defaults = {}

    config_params = {
        "interactive": args.interactive or config_defaults.get("interactive", False),
        "max_refinement_rounds": args.max_rounds or config_defaults.get("max_refinement_rounds", 5),
        "model_name": config_defaults.get("model_name"),
        "preserve_artifacts": config_defaults.get("preserve_artifacts", True),
        "emit_steps": config_defaults.get("emit_steps", True),
        "display_char_limit": config_defaults.get("display_char_limit", 1200),
    }
    config_params = {k: v for k, v in config_params.items() if v is not None}

    config = LGConfig(**config_params)
    if not config.model_name:
        parser.error("Model name must be specified via config file.")

    query = args.query or config_defaults.get("query")
    data_files = args.data_files or config_defaults.get("data_files")
    if not (data_files and query):
        parser.error("--data-files and --query are required for a new run.")

    agent = LangGraphDSStar(config)
    result = agent.run(query, data_files)
    print(f"\n{'='*60}")
    print(f"RUN COMPLETED: {result['run_id']}")
    print(f"OUTPUT: {result['output_file']}")
    print(f"FINAL RESULT:\n{result['final_result']}")
    print(f"{'='*60}")


if __name__ == "__main__":

    main()
