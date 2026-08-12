# src/features/parallel_tools.py

import json
import time
import threading
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
    print_table
)
from src.utils.menu import get_user_input


# ✅ Simulated Tools (same as tool_calling but extended)

def tool_get_weather(city: str) -> dict:
    """Simulated weather tool"""
    time.sleep(0.5)  # Simulate API delay
    weather_data = {
        "London": {
            "temp": "15°C",
            "condition": "Cloudy",
            "humidity": "80%",
            "wind": "12 km/h"
        },
        "New York": {
            "temp": "22°C",
            "condition": "Sunny",
            "humidity": "60%",
            "wind": "8 km/h"
        },
        "Tokyo": {
            "temp": "28°C",
            "condition": "Humid",
            "humidity": "85%",
            "wind": "5 km/h"
        },
        "Paris": {
            "temp": "18°C",
            "condition": "Rainy",
            "humidity": "75%",
            "wind": "15 km/h"
        },
        "Dubai": {
            "temp": "38°C",
            "condition": "Hot & Sunny",
            "humidity": "45%",
            "wind": "10 km/h"
        }
    }
    return weather_data.get(
        city,
        {
            "temp": "20°C",
            "condition": "Clear",
            "humidity": "65%",
            "wind": "7 km/h"
        }
    )


def tool_get_time(timezone: str) -> dict:
    """Simulated time tool"""
    time.sleep(0.5)  # Simulate API delay
    times = {
        "UTC": {"time": "14:30:00", "offset": "+0"},
        "EST": {"time": "09:30:00", "offset": "-5"},
        "PST": {"time": "06:30:00", "offset": "-8"},
        "IST": {"time": "20:00:00", "offset": "+5:30"},
        "JST": {"time": "23:30:00", "offset": "+9"},
        "GST": {"time": "18:30:00", "offset": "+4"},
    }
    result = times.get(
        timezone,
        {"time": "12:00:00", "offset": "unknown"}
    )
    result["timezone"] = timezone
    return result


def tool_get_stock(symbol: str) -> dict:
    """Simulated stock price tool"""
    time.sleep(0.5)  # Simulate API delay
    stocks = {
        "AAPL": {
            "price": "$182.50",
            "change": "+1.2%",
            "volume": "52M"
        },
        "GOOGL": {
            "price": "$141.80",
            "change": "-0.5%",
            "volume": "23M"
        },
        "MSFT": {
            "price": "$378.90",
            "change": "+0.8%",
            "volume": "18M"
        },
        "AMZN": {
            "price": "$178.25",
            "change": "+2.1%",
            "volume": "35M"
        },
    }
    return stocks.get(
        symbol.upper(),
        {
            "price": "$100.00",
            "change": "0%",
            "volume": "10M"
        }
    )


def tool_get_news(topic: str) -> dict:
    """Simulated news tool"""
    time.sleep(0.5)  # Simulate API delay
    news = {
        "AI": {
            "headline": "New AI Model Breaks Performance Records",
            "source": "TechNews",
            "time": "2 hours ago"
        },
        "Python": {
            "headline": "Python 4.0 Release Date Announced",
            "source": "DevWeekly",
            "time": "5 hours ago"
        },
        "iOS": {
            "headline": "Apple Releases iOS 18 Beta Update",
            "source": "AppleInsider",
            "time": "1 hour ago"
        },
        "Space": {
            "headline": "NASA Discovers New Exoplanet",
            "source": "SpaceNews",
            "time": "3 hours ago"
        },
    }
    return news.get(
        topic,
        {
            "headline": f"Latest news about {topic}",
            "source": "NewsPortal",
            "time": "Just now"
        }
    )


# ✅ Tools Registry
PARALLEL_TOOLS = {
    "get_weather": tool_get_weather,
    "get_time": tool_get_time,
    "get_stock": tool_get_stock,
    "get_news": tool_get_news
}


def run_tool_threaded(
    tool_name: str,
    parameters: dict,
    results: dict,
    index: int
):
    """
    Run a single tool in a thread
    Store result in shared results dict
    """
    try:
        tool_func = PARALLEL_TOOLS.get(tool_name)
        if tool_func:
            result = tool_func(**parameters)
            results[index] = {
                "tool": tool_name,
                "params": parameters,
                "result": result,
                "status": "✅ Success"
            }
        else:
            results[index] = {
                "tool": tool_name,
                "params": parameters,
                "result": {},
                "status": "❌ Tool not found"
            }
    except Exception as e:
        results[index] = {
            "tool": tool_name,
            "params": parameters,
            "result": {},
            "status": f"❌ Error: {e}"
        }


def run_tools_parallel(tool_calls: list) -> dict:
    """
    Run multiple tools at same time
    using threading

    tool_calls = [
        {"tool": "get_weather", "params": {"city": "Tokyo"}},
        {"tool": "get_time", "params": {"timezone": "JST"}},
    ]
    """
    threads = []
    results = {}

    # ✅ Create thread for each tool
    for i, tool_call in enumerate(tool_calls):
        thread = threading.Thread(
            target=run_tool_threaded,
            args=(
                tool_call["tool"],
                tool_call["params"],
                results,
                i
            )
        )
        threads.append(thread)

    # ✅ Start ALL threads at same time
    start_time = time.time()

    for thread in threads:
        thread.start()

    # ✅ Wait for ALL to finish
    for thread in threads:
        thread.join()

    end_time = time.time()
    total_time = round(end_time - start_time, 2)

    return {
        "results": results,
        "total_time": total_time,
        "tools_count": len(tool_calls)
    }


def run_tools_sequential(tool_calls: list) -> dict:
    """
    Run tools ONE by ONE (sequential)
    for comparison with parallel
    """
    results = {}
    start_time = time.time()

    for i, tool_call in enumerate(tool_calls):
        tool_func = PARALLEL_TOOLS.get(tool_call["tool"])
        if tool_func:
            result = tool_func(**tool_call["params"])
            results[i] = {
                "tool": tool_call["tool"],
                "params": tool_call["params"],
                "result": result,
                "status": "✅ Success"
            }

    end_time = time.time()
    total_time = round(end_time - start_time, 2)

    return {
        "results": results,
        "total_time": total_time,
        "tools_count": len(tool_calls)
    }


def decide_parallel_tools(user_request: str) -> list:
    """Ask AI which tools to run in parallel"""

    tools_desc = """
- get_weather: Get weather for a city | params: city (str)
- get_time: Get time for timezone | params: timezone (str)
- get_stock: Get stock price | params: symbol (str)
- get_news: Get latest news | params: topic (str)
"""

    prompt = f"""
Available tools:
{tools_desc}

User request: {user_request}

Identify ALL tools needed to fully answer this request.
These will run in PARALLEL at the same time.

Respond in this EXACT JSON format only:
{{
    "tools": [
        {{"tool": "tool_name", "params": {{"param": "value"}}}},
        {{"tool": "tool_name", "params": {{"param": "value"}}}}
    ],
    "reasoning": "why these tools are needed"
}}

Only JSON. No extra text.
"""

    response = ask_ai(
        prompt=prompt,
        system="You are an AI that identifies multiple tools needed for tasks.",
        temperature=0.1
    )

    # ✅ Clean response
    response = response.strip()
    if "```" in response:
        parts = response.split("```")
        response = parts[1] if len(parts) > 1 else response
        if response.startswith("json"):
            response = response[4:]

    try:
        data = json.loads(response)
        return data.get("tools", [])
    except json.JSONDecodeError:
        return [
            {"tool": "get_weather", "params": {"city": "London"}},
            {"tool": "get_time", "params": {"timezone": "UTC"}}
        ]


def get_combined_answer(
    user_request: str,
    all_results: dict
) -> str:
    """Ask AI to combine all tool results into one answer"""

    results_text = "\n".join([
        f"Tool {i+1} ({r['tool']}): {json.dumps(r['result'])}"
        for i, r in all_results.items()
    ])

    prompt = f"""
User asked: {user_request}

Multiple tools ran in parallel and returned:
{results_text}

Combine ALL results into one helpful,
natural language answer for the user.
Be clear and organized.
"""

    return ask_ai(
        prompt=prompt,
        system="You are a helpful assistant that combines multiple data sources.",
        temperature=0.5
    )


def run():
    """Parallel Tool Calling Feature"""

    # ✅ Header
    print_feature_header("Parallel Tool Calling")

    # ✅ Explain concept
    print_concept(
        "What is Parallel Tool Calling?",
        "Running MULTIPLE tools at the SAME TIME!\n\n"
        "Sequential (Slow):\n"
        "  Tool 1 → wait → Tool 2 → wait → Tool 3\n"
        "  Total = 0.5s + 0.5s + 0.5s = 1.5s\n\n"
        "Parallel (Fast):\n"
        "  Tool 1 ─┐\n"
        "  Tool 2 ─┼→ All done at same time!\n"
        "  Tool 3 ─┘\n"
        "  Total = 0.5s only!\n\n"
        "Available Tools:\n"
        "  🌤️  get_weather → Weather for city\n"
        "  🕐  get_time    → Time for timezone\n"
        "  📈  get_stock   → Stock prices\n"
        "  📰  get_news    → Latest news"
    )

    print_divider()

    # ✅ Demo 1: Sequential vs Parallel comparison
    print_step(
        "Demo 1",
        "Sequential vs Parallel — Speed Comparison"
    )

    tool_calls = [
        {"tool": "get_weather", "params": {"city": "Tokyo"}},
        {"tool": "get_time", "params": {"timezone": "JST"}},
        {"tool": "get_stock", "params": {"symbol": "AAPL"}},
        {"tool": "get_news", "params": {"topic": "AI"}}
    ]

    print_prompt(
        "Running 4 tools:\n"
        "  1. get_weather(Tokyo)\n"
        "  2. get_time(JST)\n"
        "  3. get_stock(AAPL)\n"
        "  4. get_news(AI)"
    )

    print_divider()

    # ✅ Sequential run
    print_step(
        "Sequential Run",
        "Running tools ONE BY ONE..."
    )

    seq_output = run_tools_sequential(tool_calls)

    print_info(
        f"Sequential completed in: "
        f"{seq_output['total_time']} seconds"
    )

    print_divider()

    # ✅ Parallel run
    print_step(
        "Parallel Run",
        "Running ALL tools AT SAME TIME..."
    )

    par_output = run_tools_parallel(tool_calls)

    print_info(
        f"Parallel completed in: "
        f"{par_output['total_time']} seconds"
    )

    print_divider()

    # ✅ Speed comparison table
    speed_diff = round(
        seq_output['total_time'] - par_output['total_time'],
        2
    )
    faster = round(
        seq_output['total_time'] / par_output['total_time'],
        1
    )

    print_table(
        title="⚡ Speed Comparison",
        columns=["Method", "Time", "Tools"],
        rows=[
            [
                "Sequential",
                f"{seq_output['total_time']}s",
                str(seq_output['tools_count'])
            ],
            [
                "Parallel",
                f"{par_output['total_time']}s",
                str(par_output['tools_count'])
            ],
            [
                "Difference",
                f"{speed_diff}s saved",
                f"{faster}x faster"
            ]
        ]
    )

    print_divider()

    # ✅ Show parallel results
    print_step(
        "Parallel Results",
        "All tool outputs collected:"
    )

    for i, result in par_output["results"].items():
        print_success(
            f"Tool {i+1}: {result['tool']} "
            f"→ {result['status']}\n"
            f"  Result: {result['result']}"
        )

    print_divider()

    # ✅ Demo 2: AI decides which tools to run
    print_step(
        "Demo 2",
        "AI decides which tools to run in parallel"
    )

    smart_request = (
        "I am traveling to Dubai tomorrow. "
        "What should I know about the weather, "
        "local time, and any AI news?"
    )

    print_prompt(f"Smart Request: {smart_request}")

    print_step("Step 1", "AI analyzing request...")
    print_thinking()

    tool_list = decide_parallel_tools(smart_request)

    print_info(
        f"AI decided to run {len(tool_list)} tools:\n" +
        "\n".join([
            f"  → {t['tool']}({t['params']})"
            for t in tool_list
        ])
    )

    print_step(
        "Step 2",
        f"Running {len(tool_list)} tools in parallel..."
    )

    smart_output = run_tools_parallel(tool_list)

    print_success(
        f"All {len(tool_list)} tools completed in "
        f"{smart_output['total_time']} seconds!"
    )

    for i, result in smart_output["results"].items():
        print_info(
            f"  {result['tool']}: {result['result']}"
        )

    print_step("Step 3", "AI combining all results...")
    print_thinking()

    combined = get_combined_answer(
        user_request=smart_request,
        all_results=smart_output["results"]
    )

    print_response(
        f"[Combined Answer from Parallel Tools]\n\n"
        f"{combined}"
    )

    print_divider()

    # ✅ Demo 3: Custom parallel request
    print_step(
        "Demo 3",
        "Try your own parallel request!"
    )

    print_info(
        "Try asking something that needs multiple tools!\n"
        "Example:\n"
        "  'Compare weather in London and Tokyo'\n"
        "  'Show me AAPL and MSFT stock prices'\n"
        "  'What time is it in IST and JST?'\n"
        "  'Show AI and Python news together'"
    )

    custom_request = get_user_input(
        "📝 Enter your request: "
    )

    if not custom_request:
        custom_request = (
            "Show me weather in London and Paris "
            "and latest AI news"
        )
        print_info(f"Using default: {custom_request}")

    print_prompt(f"Your Request: {custom_request}")

    print_step("Step 1", "AI deciding tools...")
    print_thinking()

    custom_tools = decide_parallel_tools(custom_request)

    print_info(
        f"Running {len(custom_tools)} tools in parallel:\n" +
        "\n".join([
            f"  → {t['tool']}({t['params']})"
            for t in custom_tools
        ])
    )

    print_step(
        "Step 2",
        "Running all tools simultaneously..."
    )

    custom_output = run_tools_parallel(custom_tools)

    print_success(
        f"Done in {custom_output['total_time']} seconds!"
    )

    print_step("Step 3", "Combining results...")
    print_thinking()

    custom_combined = get_combined_answer(
        user_request=custom_request,
        all_results=custom_output["results"]
    )

    print_response(
        f"[Your Parallel Request Answer]\n\n"
        f"{custom_combined}"
    )

    print_divider()

    # ✅ Summary
    print_concept(
        "Parallel Tool Calling Summary",
        "Key Takeaways:\n\n"
        "Sequential:\n"
        "  ❌ Tools run one by one\n"
        "  ❌ Slow for multiple tasks\n"
        "  ❌ Time = sum of all tools\n\n"
        "Parallel:\n"
        "  ✅ Tools run simultaneously\n"
        "  ✅ Much faster\n"
        "  ✅ Time = slowest tool only\n\n"
        "Real World Uses:\n"
        "  ✅ Dashboard data loading\n"
        "  ✅ Multi-city weather apps\n"
        "  ✅ Portfolio stock tracking\n"
        "  ✅ News aggregators\n"
        "  ✅ Travel planning apps"
    )