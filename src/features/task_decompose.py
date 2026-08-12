# src/features/task_decompose.py

import json
from src.ai_client import ask_ai
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_response,
    print_step,
    print_thinking,
    print_info,
    print_divider,
    print_prompt,
    print_success,
    print_table,
    print_error
)
from src.utils.menu import get_user_input


def decompose_task(task: str) -> list:
    """
    Ask AI to break a complex task
    into smaller sub-tasks
    """

    prompt = f"""
Break this complex task into small, simple steps.

Task: {task}

Rules:
- Each step should be simple and clear
- Steps should be in logical order
- Each step should be independently doable
- Include 4-8 steps
- Each step should have a clear deliverable

Respond in this EXACT JSON format only:
{{
    "original_task": "the task",
    "total_steps": 5,
    "steps": [
        {{
            "step_number": 1,
            "title": "Step title",
            "description": "What to do in this step",
            "deliverable": "What this step produces"
        }}
    ]
}}

Only JSON. No extra text.
"""

    response = ask_ai(
        prompt=prompt,
        system=(
            "You are an expert project manager who "
            "breaks complex tasks into simple steps."
        ),
        temperature=0.3
    )

    # ✅ Clean response
    response = response.strip()
    if "```" in response:
        parts = response.split("```")
        response = parts[1] if len(parts) > 1 else response
        if response.startswith("json"):
            response = response[4:]

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "original_task": task,
            "total_steps": 1,
            "steps": [
                {
                    "step_number": 1,
                    "title": "Complete task",
                    "description": task,
                    "deliverable": "Final result"
                }
            ]
        }


def execute_step(
    step: dict,
    original_task: str,
    previous_results: list
) -> str:
    """
    Execute a single decomposed step
    using AI with context of previous steps
    """

    previous_context = ""
    if previous_results:
        previous_context = (
            "\n\nPrevious steps completed:\n" +
            "\n".join([
                f"Step {i+1}: {r['title']}\n"
                f"Result: {r['result'][:200]}..."
                for i, r in enumerate(previous_results)
            ])
        )

    prompt = f"""
You are working on this overall task: {original_task}

Current Step: Step {step['step_number']} - {step['title']}
Description: {step['description']}
Expected Deliverable: {step['deliverable']}
{previous_context}

Complete this step. Provide a clear, detailed result
for this specific step only.
"""

    return ask_ai(
        prompt=prompt,
        system=(
            "You are an expert who completes tasks "
            "step by step with high quality output."
        ),
        temperature=0.5
    )


def compare_with_without(task: str):
    """
    Show difference between asking AI directly
    vs using task decomposition
    """

    # ✅ Without decomposition
    print_step(
        "Without Decomposition",
        "Asking AI the whole task at once..."
    )

    print_prompt(
        f"Direct Request (No Decomposition):\n"
        f"{task}"
    )
    print_thinking()

    direct_response = ask_ai(
        prompt=task,
        system="You are a helpful assistant.",
        temperature=0.7
    )

    print_response(
        f"[Without Decomposition]\n\n{direct_response}"
    )

    return direct_response


def run():
    """Task Decomposition Feature"""

    # ✅ Header
    print_feature_header("Task Decomposition")

    # ✅ Explain concept
    print_concept(
        "What is Task Decomposition?",
        "Breaking a BIG complex task into\n"
        "SMALL manageable steps.\n\n"
        "Why?\n"
        "  ❌ AI struggles with huge complex tasks\n"
        "  ✅ AI excels at small focused tasks\n\n"
        "Process:\n"
        "  1. Give AI a complex task\n"
        "  2. AI breaks it into sub-tasks\n"
        "  3. Execute each sub-task one by one\n"
        "  4. Combine results into final output\n\n"
        "Benefits:\n"
        "  ✅ Better quality per step\n"
        "  ✅ Easier to debug\n"
        "  ✅ Each step builds on previous\n"
        "  ✅ More organized output"
    )

    print_divider()

    # ✅ Demo 1: Direct vs Decomposed
    print_step(
        "Demo 1",
        "Direct Request vs Decomposed Request"
    )

    demo_task = (
        "Create a complete business plan for "
        "a mobile app that helps people learn "
        "new languages using AI"
    )

    print_info(
        f"Complex Task:\n"
        f"'{demo_task}'\n\n"
        f"First lets see what happens WITHOUT "
        f"decomposition..."
    )

    print_divider()

    # ✅ Without decomposition
    compare_with_without(demo_task)

    print_info(
        "Direct response is okay but:\n"
        "  ❌ Not detailed enough\n"
        "  ❌ Missing important parts\n"
        "  ❌ Not well structured\n\n"
        "Now lets DECOMPOSE the same task!"
    )

    print_divider()

    # ✅ With decomposition
    print_step(
        "With Decomposition",
        "Breaking task into steps first..."
    )

    print_thinking()

    decomposed = decompose_task(demo_task)

    # ✅ Show decomposed steps
    print_success(
        f"Task broken into "
        f"{decomposed.get('total_steps', len(decomposed.get('steps', [])))} steps!"
    )

    steps = decomposed.get("steps", [])

    # ✅ Display steps table
    step_rows = []
    for step in steps:
        step_rows.append([
            str(step.get("step_number", "")),
            step.get("title", ""),
            step.get("deliverable", "")
        ])

    print_table(
        title="📋 Decomposed Steps",
        columns=["Step", "Title", "Deliverable"],
        rows=step_rows
    )

    print_divider()

    # ✅ Execute each step
    print_step(
        "Executing Steps",
        "Now completing each step with AI..."
    )

    previous_results = []

    for step in steps:
        print_divider()

        print_step(
            f"Step {step.get('step_number', '?')}",
            step.get('title', 'Unknown')
        )

        print_info(
            f"Description: {step.get('description', '')}\n"
            f"Deliverable: {step.get('deliverable', '')}"
        )

        print_thinking()

        step_result = execute_step(
            step=step,
            original_task=demo_task,
            previous_results=previous_results
        )

        print_response(
            f"[Step {step.get('step_number', '?')} Result]\n\n"
            f"{step_result}"
        )

        previous_results.append({
            "title": step.get("title", ""),
            "result": step_result
        })

        print_success(
            f"Step {step.get('step_number', '?')} completed! ✅"
        )

    print_divider()

    # ✅ Combine all results
    print_step(
        "Final Combination",
        "Combining all step results into final output..."
    )

    combine_prompt = f"""
Original task: {demo_task}

Here are the results from each step:

{chr(10).join([
    f"Step {i+1} ({r['title']}):{chr(10)}{r['result']}"
    for i, r in enumerate(previous_results)
])}

Combine all these step results into one 
cohesive, well-organized final output.
Make it flow naturally as one complete document.
"""

    print_thinking()

    final_output = ask_ai(
        prompt=combine_prompt,
        system=(
            "You are an expert at combining multiple "
            "sections into one cohesive document."
        ),
        temperature=0.5
    )

    print_response(
        f"[Final Combined Output]\n\n{final_output}"
    )

    print_success("Task decomposition complete! ✅")

    print_divider()

    # ✅ Demo 2: Custom task
    print_step(
        "Demo 2",
        "Try YOUR own complex task!"
    )

    print_info(
        "Example complex tasks:\n"
        "  • 'Plan a 7 day trip to Japan'\n"
        "  • 'Create a marketing strategy for startup'\n"
        "  • 'Design a learning path for Python'\n"
        "  • 'Write a complete blog post about AI'"
    )

    custom_task = get_user_input(
        "📝 Enter your complex task: "
    )

    if not custom_task:
        custom_task = (
            "Plan a complete 5 day trip to Japan "
            "including flights, hotels, and activities"
        )
        print_info(f"Using default: {custom_task}")

    print_divider()

    # ✅ Decompose custom task
    print_step(
        "Step 1",
        "Decomposing your task..."
    )

    print_thinking()

    custom_decomposed = decompose_task(custom_task)
    custom_steps = custom_decomposed.get("steps", [])

    print_success(
        f"Broken into {len(custom_steps)} steps!"
    )

    custom_step_rows = []
    for step in custom_steps:
        custom_step_rows.append([
            str(step.get("step_number", "")),
            step.get("title", ""),
            step.get("deliverable", "")
        ])

    print_table(
        title="📋 Your Decomposed Steps",
        columns=["Step", "Title", "Deliverable"],
        rows=custom_step_rows
    )

    print_divider()

    # ✅ Execute custom steps
    print_step(
        "Step 2",
        "Executing each step..."
    )

    custom_previous = []

    for step in custom_steps:
        print_divider()

        print_step(
            f"Step {step.get('step_number', '?')}",
            step.get('title', '')
        )

        print_thinking()

        result = execute_step(
            step=step,
            original_task=custom_task,
            previous_results=custom_previous
        )

        print_response(
            f"[Step {step.get('step_number', '?')}]\n\n"
            f"{result}"
        )

        custom_previous.append({
            "title": step.get("title", ""),
            "result": result
        })

        print_success(
            f"Step {step.get('step_number', '?')} done! ✅"
        )

    print_divider()

    # ✅ Final combination
    print_step(
        "Step 3",
        "Combining everything..."
    )

    custom_combine = f"""
Original task: {custom_task}

Results from each step:

{chr(10).join([
    f"Step {i+1} ({r['title']}):{chr(10)}{r['result']}"
    for i, r in enumerate(custom_previous)
])}

Combine into one complete, organized final output.
"""

    print_thinking()

    custom_final = ask_ai(
        prompt=custom_combine,
        system=(
            "You are an expert at combining sections "
            "into one cohesive document."
        ),
        temperature=0.5
    )

    print_response(
        f"[Your Final Output]\n\n{custom_final}"
    )

    print_success("Your task decomposition complete! ✅")

    print_divider()

    # ✅ Summary
    print_concept(
        "Task Decomposition Summary",
        "Key Takeaways:\n\n"
        "Process:\n"
        "  Complex Task\n"
        "      ↓\n"
        "  Break into Sub-tasks\n"
        "      ↓\n"
        "  Execute Each Step\n"
        "      ↓\n"
        "  Combine Results\n"
        "      ↓\n"
        "  Final Output\n\n"
        "Benefits:\n"
        "  ✅ Better quality per step\n"
        "  ✅ AI handles small tasks better\n"
        "  ✅ Each step builds on previous\n"
        "  ✅ Easier to debug & improve\n"
        "  ✅ More detailed output\n\n"
        "When to Use:\n"
        "  ✅ Complex multi-part tasks\n"
        "  ✅ Tasks needing research + writing\n"
        "  ✅ Planning & strategy tasks\n"
        "  ✅ Multi-step processes\n\n"
        "When NOT to Use:\n"
        "  ❌ Simple one-line questions\n"
        "  ❌ Quick factual lookups"
    )