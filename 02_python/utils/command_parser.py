# utils/command_parser.py
import logging
from typing import Dict, Any

"""
TODO 10-1: Add a logger object.
Log command name, options, and arguments.

Logging format:
f"Command name: {command_name}"
f"Options: {options}"
f"Positional args: {positional_args}"
"""
logger = logging.getLogger(__name__)


class CommandParser:
    """
    A class for parsing commands and extracting command name, options, and arguments.

    Args:
        verbose (bool, optional): If True, print additional information during parsing. Defaults to False.

    Methods:
        parse_command(input_command: str) -> Dict[str, Any]:
            Parses the input command and returns a dictionary containing the command name, options, and arguments.

    """

    def __init__(self, verbose: bool = False) -> None:
        # TODO 2-1: Initialize the verbose attribute.
        self.verbose = verbose

    def parse_command(self, input_command: str) -> Dict[str, Any]:
        """
        Parses the input command and returns a dictionary containing the command name, options, and positional arguments.

        Args:
            input_command (str): The input command to be parsed.

        Returns:
            dict: A dictionary containing the parsed command information with the following keys:
                - 'command_name': The name of the command.
                - 'options': A list of options specified in the command.
                - 'args': A list of positional arguments specified in the command.
        """

        # TODO 2-2: Remove the following line after implementing the command parsing logic.
        command_name, options, positional_args = '', [], []
        
        command_list = input_command.split()
        command_name = command_list[0]
        
        for command in command_list[1:]:
            if command.startswith('-'): 
                options.append(command)
            else:
                positional_args.append(command)

        if self.verbose:
            print(f"Command name: {command_name}, Options: {options}, Positional args: {positional_args}")

        logger.info(f"Command name: {command_name}")
        logger.info(f"Options: {options}")
        logger.info(f"Positional args: {positional_args}")

        return {
            'command_name': command_name,
            'options': options,
            'args': positional_args
        }

