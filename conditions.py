from __future__ import annotations

from dataclasses import dataclass

import config
from llm_client import LlamaCppClient

FINANCIAL_INTERVENTION_PROMPT = (
    "Verify the selected financial formula, input values, signs, units, "
    "percentage conversion and rounding. Then provide the final answer."
)

GENERIC_INTERVENTION_PROMPT = (
    "Double-check your reasoning above for mistakes before finalizing your answer."
)

# Phrasen, die im Reasoning-Text als "Warnsignal" fuer Bedingung F gelten
# (Modell zeigt Unsicherheit / beginnt eine Selbstkorrektur)
WARNING_SIGNALS = [
    "wait", "let me reconsider", "actually,", "hmm", "on second thought",
    "let me double-check", "i think i made", "that's not right", "i made a mistake",
]


def build_system_prompt() -> str:
    return (
        "You are a financial analysis assistant. Solve the given problem and "
        "provide a single final numeric answer in exactly this format:\n"
        "Final Answer: <number>"
    )


def build_prompt(task_text: str, thinking: bool) -> str:
    """Chat-Prompt

    thinking=True  -> Modell bekommt einen offenen <think>-Block (Bedingungen B-F)
    thinking=False -> kein Denkblock, direkte Antwort (Bedingung A)
    """
    system = build_system_prompt()
    assistant_prefix = "<think>\n" if thinking else "<think>\n\n</think>\n\n"

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{task_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )


@dataclass
class RunResult:
    condition: str
    final_text: str
    reasoning_tokens: int
    total_tokens: int
    total_time_seconds: float
    extensions_used: int = 0
    raw_reasoning: str = ""
    reasoning_stop: str = ""
    final_stop: str = ""
    reasoning_tokens_method: str = "server"
    safety_limit_reached: bool = False
    extension_triggers: str = ""


def _force_final_answer(client: LlamaCppClient, prompt_so_far: str,
                         forcing_text: str) -> tuple:
    is_intervention = len(forcing_text.strip()) > len("Final Answer:") + 5
    budget = config.FINAL_ANSWER_BUDGET_INTERVENTION if is_intervention else config.FINAL_ANSWER_BUDGET

    # The stop string is excluded by llama.cpp, so close the open block ourselves.
    # Interventions remain within the assistant continuation, before the closing tag.
    instruction = forcing_text.removesuffix("Final Answer:").strip()
    forced_prompt = prompt_so_far + "\n" + instruction + "\n</think>\n\nFinal Answer:"
    result = client.complete(
        forced_prompt,
        n_predict=budget,
        stop=["<|im_end|>"],
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
    )
    # /completion only returns the suffix; retain the supplied answer marker.
    return "Final Answer:" + result["text"], result["tokens_predicted"], result["time_seconds"], result.get("stop_type", "")


def run_condition_a(client: LlamaCppClient, task_text: str) -> RunResult:
    direct_task = (
        task_text
        + "\n\nRespond with only the final numeric result in the exact "
        "format 'Final Answer: <number>'. Do not show any calculation, "
        "formula, or explanation."
    )
    prompt = build_prompt(direct_task, thinking=False)
    result = client.complete(
        prompt, n_predict=config.FINAL_ANSWER_BUDGET,
        stop=["<|im_end|>"], temperature=config.TEMPERATURE, top_p=config.TOP_P,
    )
    return RunResult(
        condition="A_direct", final_text=result["text"],
        reasoning_tokens=0, total_tokens=result["tokens_predicted"],
        total_time_seconds=result["time_seconds"],
        final_stop=result.get("stop_type", ""),
    )


def run_condition_b(client: LlamaCppClient, task_text: str) -> RunResult:
    """B: Free Reasoning"""
    prompt = build_prompt(task_text, thinking=True)
    result = client.complete(
        prompt, n_predict=config.FREE_REASONING_BUDGET,
        stop=["<|im_end|>"], temperature=config.TEMPERATURE, top_p=config.TOP_P,
    )
    reasoning, separator, final = result["text"].partition("</think>")
    reasoning_tokens = min(client.count_tokens(reasoning), result["tokens_predicted"]) if separator else result["tokens_predicted"]
    return RunResult(
        condition="B_free", final_text=final if separator else "",
        reasoning_tokens=reasoning_tokens,
        total_tokens=result["tokens_predicted"],
        total_time_seconds=result["time_seconds"],
        raw_reasoning=reasoning,
        reasoning_stop="word" if separator else result.get("stop_type", ""),
        final_stop=result.get("stop_type", ""),
        reasoning_tokens_method="retokenized" if separator else "server",
        safety_limit_reached=result["stopped_limit"],
    )


def _run_fixed_budget(client: LlamaCppClient, task_text: str, budget: int,
                       forcing_text: str, condition_label: str) -> RunResult:
    """Bedingungen C, D, E (festes Budget + Forcing)"""
    prompt = build_prompt(task_text, thinking=True)
    reasoning = client.complete(
        prompt, n_predict=budget,
        stop=["</think>", "<|im_end|>"],
        temperature=config.TEMPERATURE, top_p=config.TOP_P,
    )

    prompt_after_reasoning = prompt + reasoning["text"]
    total_tokens = reasoning["tokens_predicted"]
    total_time = reasoning["time_seconds"]

    final_text, forced_tokens, forced_time, final_stop = _force_final_answer(
        client, prompt_after_reasoning, forcing_text
    )

    total_tokens += forced_tokens
    total_time += forced_time

    return RunResult(
        condition=f"{condition_label}_budget{budget}",
        final_text=final_text,
        reasoning_tokens=reasoning["tokens_predicted"],
        total_tokens=total_tokens,
        total_time_seconds=total_time,
        raw_reasoning=reasoning["text"],
        reasoning_stop=reasoning.get("stop_type", ""), final_stop=final_stop,
    )


def run_condition_c(client: LlamaCppClient, task_text: str, budget: int) -> RunResult:
    """C: Fixed Budget, neutrales Forcing."""
    return _run_fixed_budget(client, task_text, budget, "\nFinal Answer:", "C_fixed")


def run_condition_d(client: LlamaCppClient, task_text: str, budget: int) -> RunResult:
    """D: Fixed Budget + generische Pruefaufforderung."""
    forcing = f"\n{GENERIC_INTERVENTION_PROMPT}\nFinal Answer:"
    return _run_fixed_budget(client, task_text, budget, forcing, "D_generic")


def run_condition_e(client: LlamaCppClient, task_text: str, budget: int) -> RunResult:
    """E: Fixed Budget + finanzspezifische Pruefaufforderung."""
    forcing = f"\n{FINANCIAL_INTERVENTION_PROMPT}\nFinal Answer:"
    return _run_fixed_budget(client, task_text, budget, forcing, "E_financial")


def run_condition_f(client: LlamaCppClient, task_text: str) -> RunResult:
    """F: Adaptive Strategy """
    prompt = build_prompt(task_text, thinking=True)
    budget = config.ADAPTIVE_START_BUDGET
    extensions_used = 0
    total_tokens = 0
    total_time = 0.0
    accumulated_reasoning = ""
    extension_triggers = []

    while True:
        reasoning = client.complete(
            prompt + accumulated_reasoning, n_predict=budget,
            stop=["</think>", "<|im_end|>"],
            temperature=config.TEMPERATURE, top_p=config.TOP_P,
        )
        accumulated_reasoning += reasoning["text"]
        total_tokens += reasoning["tokens_predicted"]
        total_time += reasoning["time_seconds"]

        needs_extension = (
            reasoning["stopped_limit"]
            and extensions_used < config.ADAPTIVE_MAX_EXTENSIONS
            and _contains_warning_signal(reasoning["text"])
        )
        if needs_extension:
            extension_triggers.extend(signal for signal in WARNING_SIGNALS if signal in reasoning["text"].lower())
            budget = config.ADAPTIVE_EXTENSION
            extensions_used += 1
            continue
        break

    final_text, forced_tokens, forced_time, final_stop = _force_final_answer(
        client, prompt + accumulated_reasoning, "\nFinal Answer:"
    )
    total_tokens += forced_tokens
    total_time += forced_time

    return RunResult(
        condition="F_adaptive", final_text=final_text,
        reasoning_tokens=total_tokens - forced_tokens,
        total_tokens=total_tokens, total_time_seconds=total_time,
        extensions_used=extensions_used, raw_reasoning=accumulated_reasoning,
        reasoning_stop=reasoning.get("stop_type", ""), final_stop=final_stop,
        extension_triggers=" | ".join(sorted(set(extension_triggers))),
    )


def _contains_warning_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in WARNING_SIGNALS)
