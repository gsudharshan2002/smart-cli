from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import print as rprint
from rich.table import Table
import time
import random

console = Console()


def print_welcome(app_name: str):
    """Print welcome banner"""
    console.print(Panel.fit(
        f"[bold cyan]👽 Sudharshan's {app_name}[/bold cyan]\n"
        f"[dim]Powered by AI — Built with Python[/dim]",
        border_style="cyan"
    ))
    console.print()

def print_feature_header(feature_name: str):
    """Print feature title"""
    console.print()
    console.print(Rule(
        f"[bold yellow] {feature_name}[/bold yellow]",
        style="yellow"
    ))
    console.print()

def print_concept(title: str, explanation: str):
    """Print concept explanation box"""
    console.print(Panel(
        f"[white]{explanation}[/white]",
        title=f"[bold blue] {title}[/bold blue]",
        border_style="blue"
    ))
    console.print()

def print_prompt(prompt: str):
    """Print the prompt being sent"""
    console.print(Panel(
        f"[bold green]{prompt}[/bold green]",
        title="[bold] Prompt Sent to AI[/bold]",
        border_style="green"
    ))
    console.print()

def print_response(response: str):
    """Print AI response"""
    console.print(Panel(
        f"[white]{response}[/white]",
        title="[bold magenta] AI Response[/bold magenta]",
        border_style="magenta"
    ))
    console.print()

def print_step(step: str, description: str):
    """Print step info"""
    console.print(
        f"[bold cyan]▶ {step}[/bold cyan] → [white]{description}[/white]"
    )

def print_success(message: str):
    """Print success message"""
    console.print(f"[bold green] {message}[/bold green]")

def print_error(message: str):
    """Print error message"""
    console.print(f"[bold red] {message}[/bold red]")


def print_info(message: str):
    """Print info message"""
    console.print(f"[bold yellow] {message}[/bold yellow]")


def print_divider():
    """Print divider line"""
    console.print(Rule(style="dim"))


def print_table(title: str, columns: list, rows: list):
    """Print a table"""
    table = Table(title=title, border_style="cyan")

    for col in columns:
        table.add_column(col, style="bold white")

    for row in rows:
        table.add_row(*row)

    console.print(table)
    console.print()


def print_thinking():
    """Animated 'thinking' spinner (like real AI chat UIs)"""
    with console.status(
        "[dim] Sudharshan is thinking...[/dim]",
        spinner="dots"
    ):
        time.sleep(random.uniform(1.0, 2.0))


def type_response(response: str, title: str = "AI Response", delay: float = 0.015):
    """
    Stream the response character-by-character, like a typewriter
    (with a natural extra pause after punctuation)
    """
    console.print(f"\n[bold magenta] {title}[/bold magenta]")
    console.print(Rule(style="magenta"))

    for char in response:
        console.print(char, end="", markup=False, highlight=False)

        if char in ".!?":
            time.sleep(delay * 18)
        elif char in ",;:\n":
            time.sleep(delay * 8)
        else:
            time.sleep(delay + random.uniform(0, delay))

    console.print("\n")
    console.print(Rule(style="magenta"))
    console.print()


def print_bye():
    """Print goodbye message"""
    console.print()
    console.print(Panel.fit(
        "[bold cyan] Thanks for using Smart CLI![/bold cyan]\n"
        "[dim]See you next time![/dim]",
        border_style="cyan"
    ))
