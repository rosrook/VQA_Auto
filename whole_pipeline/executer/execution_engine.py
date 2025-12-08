"""
Execution engine: build prompts per (sample, agent) and optionally run agents.

Inputs:
- VQA data file: .json (list) or .jsonl (one JSON per line)
- Agent plan file: {"agents": [{"name": ..., "prompt" or "prompt_scaffold": ...}, ...]}
- Agent config file: {"agents": [{"name": ..., "type": "openai|local_hf|remote_hf", "params": {...}}]}

Outputs (default: ./run_outputs):
- tasks.jsonl      : one line per (sample, agent) with constructed prompt
- results.jsonl    : (when --run) model outputs per task
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_adapters import (
    BaseAgent,
    OpenAIModelAgent,
    LocalHFModelAgent,
    RemoteHFModelAgent,
)

# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        data: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    with path.open("r", encoding="utf-8") as f:
        content = json.load(f)
        if isinstance(content, list):
            return content
        if isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
            return content["data"]
        raise ValueError("Unsupported JSON structure; expected list or {data: list}.")


def load_agents(plan_path: Path) -> List[Dict[str, Any]]:
    if not plan_path.is_file():
        raise FileNotFoundError(f"Agent plan not found: {plan_path}")
    with plan_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    agents = plan.get("agents")
    if not isinstance(agents, list):
        raise ValueError("Agent plan must contain 'agents' as a list.")
    for a in agents:
        if "prompt" not in a and "prompt_scaffold" not in a:
            raise ValueError(f"Agent missing prompt/prompt_scaffold: {a}")
    return agents


def load_agent_config(cfg_path: Path) -> Dict[str, BaseAgent]:
    """
    读取 agent_config.json，构建 name -> Agent 实例映射
    预期结构:
    {
      "agents": [
        {"name": "reasoning_agent", "type": "openai", "params": {...}},
        {"name": "local_agent", "type": "local_hf", "params": {...}},
        ...
      ]
    }
    """
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Agent config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    items = cfg.get("agents")
    if not isinstance(items, list):
        raise ValueError("agent_config must contain 'agents' list")

    agents: Dict[str, BaseAgent] = {}
    for item in items:
        name = item.get("name")
        a_type = item.get("type")
        params = item.get("params", {}) or {}
        if not name or not a_type:
            raise ValueError(f"agent config item missing name/type: {item}")

        if a_type == "openai":
            agent = OpenAIModelAgent(**params)
        elif a_type == "local_hf":
            agent = LocalHFModelAgent(**params)
        elif a_type == "remote_hf":
            agent = RemoteHFModelAgent(**params)
        else:
            raise ValueError(f"unsupported agent type: {a_type}")

        agents[name] = agent
    return agents


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def truncate_str(text: str, max_len: int = 2000) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[: max_len // 2] + "\n...\n" + text[-max_len // 2 :]


def make_agent_prompt(agent: Dict[str, Any], sample: Dict[str, Any]) -> str:
    body = agent.get("prompt") or agent.get("prompt_scaffold") or ""
    context = truncate_str(json.dumps(sample, ensure_ascii=False), max_len=2000)
    return (
        f"{body}\n\n"
        f"Context (JSON, truncated if long):\n"
        f"{context}\n\n"
        f"Return structured JSON with your findings."
    )


# --------------------------------------------------------------------------- #
# Main execution: build tasks
# --------------------------------------------------------------------------- #
def build_tasks(samples: List[Dict[str, Any]], agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for sample in samples:
        sample_id = sample.get("sample_id") or sample.get("id") or sample.get("qid") or "unknown"
        for agent in agents:
            tasks.append(
                {
                    "sample_id": sample_id,
                    "agent": agent.get("name", "unknown_agent"),
                    "prompt": make_agent_prompt(agent, sample),
                }
            )
    return tasks


def save_tasks(tasks: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"tasks.jsonl written to: {tasks_path}")


# --------------------------------------------------------------------------- #
# Execute tasks with agents
# --------------------------------------------------------------------------- #
def execute_tasks(
    tasks_path: Path,
    agents_map: Dict[str, BaseAgent],
    output_dir: Path,
) -> None:
    results_path = output_dir / "results.jsonl"
    with tasks_path.open("r", encoding="utf-8") as f_in, results_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            task = json.loads(line)
            agent_name = task.get("agent")
            prompt = task.get("prompt", "")
            if agent_name not in agents_map:
                # 跳过未知 agent
                continue
            agent = agents_map[agent_name]
            try:
                resp = agent.generate(prompt=prompt)
                out = {
                    "sample_id": task.get("sample_id"),
                    "agent": agent_name,
                    "text": resp.get("text"),
                    "raw": resp.get("raw"),
                    "usage": resp.get("usage"),
                }
            except Exception as e:
                out = {
                    "sample_id": task.get("sample_id"),
                    "agent": agent_name,
                    "error": str(e),
                }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"results.jsonl written to: {results_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Execution engine: build tasks and optionally run agents.")
    parser.add_argument("vqa_file", help="待筛选数据文件 (.json 或 .jsonl)")
    parser.add_argument("agent_plan", help="agent计划文件 (JSON, 包含 agents 列表)")
    parser.add_argument("--agent-config", help="agent配置文件 (JSON, 用于实例化具体模型适配器)")
    parser.add_argument("--output-dir", default="run_outputs", help="输出目录 (默认: run_outputs)")
    parser.add_argument("--run", action="store_true", help="指定后会调用 agent 执行并输出 results.jsonl")
    args = parser.parse_args()

    vqa_path = Path(args.vqa_file)
    agent_path = Path(args.agent_plan)
    output_dir = Path(args.output_dir)

    samples = load_json_or_jsonl(vqa_path)
    agents = load_agents(agent_path)
    tasks = build_tasks(samples, agents)
    save_tasks(tasks, output_dir)

    if args.run:
        if not args.agent_config:
            raise ValueError("需要提供 --agent-config 才能执行模型调用")
        agents_map = load_agent_config(Path(args.agent_config))
        tasks_path = output_dir / "tasks.jsonl"
        execute_tasks(tasks_path, agents_map, output_dir)


if __name__ == "__main__":
    main()

