from .base_command import BaseCommand
import os
import shutil
from typing import List

class MoveCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the MoveCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Move a file or directory to another location'
        self.usage = 'Usage: mv [source] [destination]'

        # TODO 5-1: Initialize any additional attributes you may need.
        # Refer to list_command.py, grep_command.py to implement this.
        # ...
        self.name = 'mv'
        self.options = options
        # args = ['qwe.py', 'commands']
        self.file = args[0] if args else ''
        self.directory = args[1] if len(args) > 1 else ''

    def execute(self) -> None:
        """
        Execute the move command.
        Supported options:
            -i: Prompt the user before overwriting an existing file.
            -v: Enable verbose mode (print detailed information)
        
        TODO 5-2: Implement the functionality to move a file or directory to another location.
        You may need to handle exceptions and print relevant error messages.
        """
        # Your code here
        overwrite = '-i' in self.options
        moving_info = '-v' in self.options

        try:
            destination_path = os.path.join(self.directory, self.file)

            if moving_info:
                print('mv: moving', self.file, 'to', self.directory)

            if overwrite and self.file_exists(self.directory, self.file):
                print(f"mv: overwrite '{self.directory}/{self.file}'? (y/n)")
                answer = input()
                if answer != 'y':
                    return
                else:
                    # 복제
                    shutil.move(self.file, destination_path)
                    return 
            
            # 이동
            shutil.move(self.file, self.directory)

        # 이동하려는 폴더에 같은 이름의 파일이 있으면 발생하는 에러 처리
        except shutil.Error:
            print(f"mv: cannot move '{self.file}' to '{self.directory}': Destination path {destination_path} already exists")
        
        # 그 이외의 에러 처리 
        except Exception:
            print(f"mv: cannot move '{self.file}' to '{self.directory}'")
    
    def file_exists(self, directory: str, file_name: str) -> bool:
        """
        Check if a file exists in a directory.
        Feel free to use this method in your execute() method.

        Args:
            directory (str): The directory to check.
            file_name (str): The name of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        file_path = os.path.join(directory, file_name)
        return os.path.exists(file_path)