from .base_command import BaseCommand
import os
import shutil
from typing import List

class ChangeDirectoryCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the ChangeDirectoryCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Change the current working directory'
        self.usage = 'Usage: cd [options] [directory]'

        # TODO 7-1: Initialize any additional attributes you may need.
        # Refer to list_command.py, grep_command.py to implement this.
        # ...
        self.name = 'cd'
        self.options = options
        self.directory = args[0] if args else ''

    def execute(self) -> None:
        """
        Execute the cd command.
        Supported options:
            -v: Enable verbose mode (print detailed information)
        
        TODO 7-2: Implement the functionality to change the current working directory.
        You may need to handle exceptions and print relevant error messages.
        """
        # Your code here
        info = '-v' in self.options
        
        try:
            if not self.directory:
                bins, target_path = os.path.split(os.getcwd())
                os.chdir(os.pardir)
            
            else:
                target_path = self.directory
                os.chdir(self.directory)

            if info:
                print(f"cd: changing directory to '{target_path}'")
            
            BaseCommand.current_path = os.getcwd()
        
        except FileNotFoundError as e:
            print(f"cd: cannot change directory to '{self.directory}': [Errno 2] No such file or directory: '{self.directory}'")

        except Exception as e:
            print(f"another error: {e}")
