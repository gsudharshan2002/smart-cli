from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


FEATURES = {
    "1": "Sampling & Temperature",
    "2": "Top-K & Top-P",
    "3": "Grounding",
    "4": "Prompt Anatomy",
    "5": "Self Consistency",
    "6": "Tool Calling",
    "7": "Parallel Tool Calling",
    "8": "System Role Logic",
    "9": "Zero Shot & Few Shot",
    "10": "Task Decomposition",
    "11": "Chain of Thoughts (COT)",
    "12": "Pydantic Instructor",
    "13": "rag",
    "0": "Exit"
}


def show_menu():
    """Display the main menu"""
    table = Table(
        title="[bold cyan]🤖 Smart CLI — Choose a Feature[/bold cyan]",
        border_style="cyan",
        show_header=True,
        header_style="bold yellow"
    )

    table.add_column("Option", justify="center", style="bold green")
    table.add_column("Feature", style="white")

    for key, name in FEATURES.items():
        if key == "0":
            table.add_row("[red]0[/red]", "[red]Exit[/red]")
        else:
            table.add_row(key, name)

    console.print()
    console.print(table)
    console.print()

def get_choice() -> str:
    """Get user menu choice"""
    choice = console.input(
        "[bold cyan] Enter your choice: [/bold cyan]"
    ).strip()
    return choice


def get_user_input(prompt_text: str) -> str:
    """Get custom input from user"""
    user_input = console.input(
        f"[bold green]{prompt_text}[/bold green]"
    ).strip()
    return user_input


def press_enter_to_continue():
    """Wait for user to continue"""
    console.print()
    console.input("[dim]Press ENTER to return to menu...[/dim]")


def is_valid_choice(choice: str) -> bool:
    """Check if choice is valid"""
    return choice in FEATURES