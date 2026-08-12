# src/features/tool_calling.py

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
    print_success
)
from src.utils.menu import get_user_input




def tool_get_weather(city: str) -> dict:
    """Simulated weather tool"""
    print_step(
        "🔧 Tool Running",
        f"get_weather(city='{city}')"
    )

    # Fake weather data
    weather_data = {
        "London": {"temp": "15°C", "condition": "Cloudy", "humidity": "80%"},
        "New York": {"temp": "22°C", "condition": "Sunny", "humidity": "60%"},
        "Tokyo": {"temp": "28°C", "condition": "Humid", "humidity": "85%"},
        "Paris": {"temp": "18°C", "condition": "Rainy", "humidity": "75%"},
    }

    result = weather_data.get(
        city,
        {"temp": "20°C", "condition": "Clear", "humidity": "65%"}
    )

    print_success(f"Tool returned: {result}")
    return result


def tool_calculate(expression: str) -> dict:
    """Simulated calculator tool"""
    print_step(
        "🔧 Tool Running",
        f"calculate(expression='{expression}')"
    )

    try:
        # Safe eval for math only
        allowed = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max
        }
        result = eval(expression, allowed)
        output = {"result": result, "expression": expression}
    except Exception:
        output = {"result": "Error", "expression": expression}

    print_success(f"Tool returned: {output}")
    return output


def tool_search_database(query: str) -> dict:
    """Simulated database search tool"""
    print_step(
        "🔧 Tool Running",
        f"search_database(query='{query}')"
    )

    # Fake database
    database = {
        "python": {
            "type": "Programming Language",
            "created": "1991",
            "creator": "Guido van Rossum",
            "use": "General purpose, AI, Web"
        },
        "javascript": {
            "type": "Programming Language",
            "created": "1995",
            "creator": "Brendan Eich",
            "use": "Web development, Frontend"
        },
        "swift": {
            "type": "Programming Language",
            "created": "2014",
            "creator": "Apple Inc",
            "use": "iOS, macOS development"
        }
    }

    key = query.lower()
    result = database.get(
        key,
        {"info": f"No data found for '{query}'"}
    )

    print_success(f"Tool returned: {result}")
    return result


def tool_get_time(timezone: str) -> dict:
    """Simulated time tool"""
    print_step(
        "🔧 Tool Running",
        f"get_time(timezone='{timezone}')"
    )

    # Fake time data
    times = {
        "UTC": "14:30:00",
        "EST": "09:30:00",
        "PST": "06:30:00",
        "IST": "20:00:00",
        "JST": "23:30:00",
    }

    result = {
        "timezone": timezone,
        "time": times.get(timezone, "12:00:00"),
        "format": "24h"
    }

    print_success(f"Tool returned: {result}")
    return result


# ✅ Available tools registry
TOOLS = {
    "get_weather": {
        "function": tool_get_weather,
        "description": "Get weather for a city",
        "parameters": ["city"],
        "example": "get_weather('London')"
    },
    "calculate": {
        "function": tool_calculate,
        "description": "Calculate math expressions",
        "parameters": ["expression"],
        "example": "calculate('25 * 4 + 10')"
    },
    "search_database": {
        "function": tool_search_database,
        "description": "Search info about programming languages",
        "parameters": ["query"],
        "example": "search_database('python')"
    },
    "get_time": {
        "function": tool_get_time,
        "description": "Get current time for timezone",
        "parameters": ["timezone"],
        "example": "get_time('IST')"
    }
}


def decide_tool(user_request: str) -> dict:
    """
    Ask AI which tool to use
    and what parameters to pass
    """

    tools_description = "\n".join([
        f"- {name}: {info['description']} | params: {info['parameters']}"
        for name, info in TOOLS.items()
    ])

    decision_prompt = f"""
You have access to these tools:
{tools_description}

User request: {user_request}

Decide which tool to use and what parameters to pass.
Respond in this EXACT JSON format only:
{{
    "tool": "tool_name_here",
    "parameters": {{"param_name": "param_value"}},
    "reasoning": "why you chose this tool"
}}

Only respond with JSON. No extra text.
"""

    response = ask_ai(
        prompt=decision_prompt,
        system="You are an AI that selects the right tool for tasks.",
        temperature=0.1
    )

    # ✅ Clean response
    response = response.strip()
    if "```" in response:
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "tool": "calculate",
            "parameters": {"expression": "1+1"},
            "reasoning": "fallback"
        }


def run_tool(tool_name: str, parameters: dict) -> dict:
    """Execute the selected tool"""

    if tool_name not in TOOLS:
        return {"error": f"Tool '{tool_name}' not found"}

    tool_func = TOOLS[tool_name]["function"]

    # ✅ Call tool with parameters
    try:
        result = tool_func(**parameters)
        return result
    except Exception as e:
        return {"error": str(e)}


def get_final_answer(
    user_request: str,
    tool_name: str,
    tool_result: dict
) -> str:
    """Ask AI to form final answer using tool result"""

    final_prompt = f"""
User asked: {user_request}

You used tool: {tool_name}
Tool returned: {json.dumps(tool_result)}

Now give a helpful natural language answer
to the user based on the tool result.
Be concise and friendly.
"""

    return ask_ai(
        prompt=final_prompt,
        system="You are a helpful assistant that uses tool results to answer users.",
        temperature=0.5
    )


def run():
    """Tool Calling Feature"""

    # ✅ Header
    print_feature_header("Tool Calling")

    # ✅ Explain concept
    print_concept(
        "What is Tool Calling?",
        "Tool Calling = AI decides which tool to use\n"
        "based on what user asks.\n\n"
        "Flow:\n"
        "  1. User asks question\n"
        "  2. AI decides which tool to use\n"
        "  3. AI extracts parameters\n"
        "  4. Tool runs & returns result\n"
        "  5. AI forms final answer\n\n"
        "Our Tools (Simulated):\n"
        "  🌤️  get_weather   → Weather for any city\n"
        "  🧮  calculate     → Math expressions\n"
        "  🔍  search_database → Programming language info\n"
        "  🕐  get_time      → Time for any timezone\n\n"
        "Note: Tools are SIMULATED (no real API calls)"
    )

    print_divider()

    # ✅ Demo requests
    demo_requests = [
        "What is the weather like in Tokyo?",
        "Calculate 150 multiplied by 7 plus 50",
        "Tell me about the Python programming language",
        "What time is it in IST timezone?"
    ]

    for i, request in enumerate(demo_requests, 1):

        print_step(
            f"Demo {i}",
            f"Request: {request}"
        )

        print_prompt(f"User Request: {request}")

        # ✅ Step 1: AI decides tool
        print_step("Step 1", "AI deciding which tool to use...")
        print_thinking()

        decision = decide_tool(request)

        print_info(
            f"AI Decision:\n"
            f"  Tool: {decision.get('tool')}\n"
            f"  Parameters: {decision.get('parameters')}\n"
            f"  Reasoning: {decision.get('reasoning')}"
        )

        # ✅ Step 2: Run the tool
        print_step("Step 2", "Running selected tool...")

        tool_result = run_tool(
            tool_name=decision.get("tool", ""),
            parameters=decision.get("parameters", {})
        )

        # ✅ Step 3: Get final answer
        print_step("Step 3", "AI forming final answer...")
        print_thinking()

        final_answer = get_final_answer(
            user_request=request,
            tool_name=decision.get("tool", ""),
            tool_result=tool_result
        )

        print_response(
            f"[Final Answer]\n\n{final_answer}"
        )

        print_divider()

    # ✅ Demo: Custom request
    print_step(
        "Try It",
        "Send your own request!"
    )

    print_info(
        "Available tools:\n"
        "  🌤️  Weather  → Ask about any city weather\n"
        "  🧮  Math     → Ask any calculation\n"
        "  🔍  Database → Ask about Python/JavaScript/Swift\n"
        "  🕐  Time     → Ask time in UTC/EST/PST/IST/JST"
    )

    custom_request = get_user_input(
        "📝 Enter your request: "
    )

    if not custom_request:
        custom_request = "What is the weather in Paris?"
        print_info(f"Using default: {custom_request}")

    print_prompt(f"Your Request: {custom_request}")

    # ✅ Full tool calling flow
    print_step("Step 1", "AI deciding tool...")
    print_thinking()
    decision = decide_tool(custom_request)

    print_info(
        f"AI chose:\n"
        f"  Tool: {decision.get('tool')}\n"
        f"  Params: {decision.get('parameters')}"
    )

    print_step("Step 2", "Running tool...")
    tool_result = run_tool(
        tool_name=decision.get("tool", ""),
        parameters=decision.get("parameters", {})
    )

    print_step("Step 3", "Forming answer...")
    print_thinking()
    final_answer = get_final_answer(
        user_request=custom_request,
        tool_name=decision.get("tool", ""),
        tool_result=tool_result
    )

    print_response(
        f"[Your Request Answer]\n\n{final_answer}"
    )

    print_divider()

    # ✅ Summary
    print_concept(
        "Tool Calling Summary",
        "Key Takeaways:\n\n"
        "Flow:\n"
        "  User Request\n"
        "      ↓\n"
        "  AI Decides Tool\n"
        "      ↓\n"
        "  Tool Executes\n"
        "      ↓\n"
        "  AI Forms Answer\n\n"
        "Real World Uses:\n"
        "  ✅ Weather apps\n"
        "  ✅ Calculator bots\n"
        "  ✅ Database queries\n"
        "  ✅ API integrations\n"
        "  ✅ Smart assistants"
    )