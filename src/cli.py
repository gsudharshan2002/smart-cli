from src.config import APP_NAME
from src.utils.printer import print_welcome, print_bye, print_error
from src.utils.menu import (
    show_menu,
    get_choice,
    is_valid_choice,
    press_enter_to_continue
)

from src.features import sampling
from src.features import top_k_top_p
from src.features import grounding
from src.features import prompt_anatomy 
from src.features import self_consistency
from src.features import tool_calling
from src.features import parallel_tools
from src.features import system_role 
from src.features import zero_few_shot 
from src.features import task_decompose
from src.features import cot
from src.features import pydantic_output
from src.features import rag_chat  



def run():
    """Main app loop"""

    print_welcome(APP_NAME)

    while True:
        show_menu()
        choice = get_choice()

        if choice == "0":
            print_bye()
            break

        if not is_valid_choice(choice):
            print_error("Invalid choice! Please try again.")
            continue

        handle_choice(choice)

        press_enter_to_continue()

def handle_choice(choice: str):
    """Route choice to correct feature"""

    if choice == "1":
        sampling.run()

    elif choice == "2":
        top_k_top_p.run()

    elif choice == "3":
        grounding.run() 

    elif choice == "4":
        prompt_anatomy.run()

    elif choice == "5":
        self_consistency.run()

    elif choice == "6":
       tool_calling.run()

    elif choice == "7":
        parallel_tools.run() 

    elif choice == "8":
        system_role.run()

    elif choice == "9":
        zero_few_shot.run()

    elif choice == "10":
        task_decompose.run()

    elif choice == "11":
        cot.run() 

    elif choice == "12":
        pydantic_output.run() 

    elif choice == "13":
        rag_chat.run() 
