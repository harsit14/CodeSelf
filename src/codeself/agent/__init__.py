"""Agent prompting, parsing, and rollout helpers."""

from codeself.agent.generation import (
    CodeGenerator,
    GenerationRequest,
    GenerationResult,
    MockGenerator,
    StaticGenerator,
    TransformersGenerator,
    make_generator,
)
from codeself.agent.loop import (
    AgentLoopConfig,
    AgentStep,
    AgentTrace,
    SelfDebugAgentLoop,
    StrategyAnalysis,
    analyze_traces,
    read_agent_traces_jsonl,
    run_agentic_tasks,
    write_agent_traces_jsonl,
    write_strategy_report,
)
from codeself.agent.parser import ParsedCompletion, ParseStatus, extract_code
from codeself.agent.prompts import (
    DIRECT_SOLUTION_TEMPLATE,
    DIRECT_WITH_PUBLIC_TESTS_TEMPLATE,
    PromptTemplate,
    get_prompt_template,
)
from codeself.agent.rollouts import (
    RolloutRecord,
    generate_rollouts,
    read_rollouts_jsonl,
    write_rollouts_jsonl,
)
from codeself.agent.self_debug_rollouts import (
    SelfDebugRolloutConfig,
    SelfDebugRolloutResult,
    generate_self_debug_rollouts,
    rollout_record_from_trace,
)
from codeself.agent.tools import AgentToolbox, ToolCall, ToolResult, infer_simple_revision

__all__ = [
    "AgentLoopConfig",
    "AgentStep",
    "AgentToolbox",
    "AgentTrace",
    "CodeGenerator",
    "DIRECT_SOLUTION_TEMPLATE",
    "DIRECT_WITH_PUBLIC_TESTS_TEMPLATE",
    "GenerationRequest",
    "GenerationResult",
    "MockGenerator",
    "ParsedCompletion",
    "ParseStatus",
    "PromptTemplate",
    "RolloutRecord",
    "SelfDebugAgentLoop",
    "SelfDebugRolloutConfig",
    "SelfDebugRolloutResult",
    "StaticGenerator",
    "StrategyAnalysis",
    "ToolCall",
    "ToolResult",
    "TransformersGenerator",
    "analyze_traces",
    "extract_code",
    "generate_rollouts",
    "generate_self_debug_rollouts",
    "get_prompt_template",
    "infer_simple_revision",
    "make_generator",
    "read_agent_traces_jsonl",
    "read_rollouts_jsonl",
    "rollout_record_from_trace",
    "run_agentic_tasks",
    "write_agent_traces_jsonl",
    "write_rollouts_jsonl",
    "write_strategy_report",
]
