# commands/base_command.py
import os
from typing import List

"""
TODO 3-1: The BaseCommand class has a show_usage method implemented, but the execute method is not 
implemented and is passed on to the child class. Think about why this difference is made.

Answer (You may write your answer in either Korean or English):
상속받는 class마다 execute method가 다르기 때문에 BaseCommand에서는 직접 구현하지 않는다. 

TODO 3-2: The update_current_path method of the BaseCommand class is slightly different from other methods. 
It has a @classmethod decorator and takes a cls argument instead of self. In Python, this is called a 
class method, and think about why it was implemented as a class method instead of a normal method.

Answer (You may write your answer in either Korean or English):
기존의 인스턴스 메서드와 다르게, 클래스 메서드는 클래스 전체에 대해서 공유된다. 즉, 특정 객체에 대한 것이 아닌 클래스 전체에서
공유되는 메서드라고 할 수 있다. 따라서 self가 아닌 cls를 사용하고, decorator @classmethod를 사용한다. 상속받는 모든 class에서
update_current_path를 사용할 수 있도록 클래스 메서드로 구현하였다. 
"""
class BaseCommand:
    """
    Base class for all commands. Each command should inherit from this class and 
    override the execute() method.
    
    For example, the MoveCommand class overrides the execute() method to implement 
    the mv command.

    Attributes:
        current_path (str): The current path. Usefull for commands like ls, cd, etc.
    """

    current_path = os.getcwd()

    @classmethod
    def update_current_path(cls, new_path: str):
        """
        Update the current path.
        You need to understand how class methods work.

        Args:
            new_path (str): The new path. (Must be an relative path)
        """
        BaseCommand.current_path = os.path.join(BaseCommand.current_path, new_path)

    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize a new instance of BaseCommand.

        Args:
            options (List[str]): The command options (e.g. -v, -i, etc.)
            args (List[str]): The command arguments (e.g. file names, directory names, etc.)
        """
        self.options = options
        self.args = args
        self.description = 'Helpful description of the command'
        self.usage = 'Usage: command [options] [arguments]'

    def show_usage(self) -> None:
        """
        Show the command usage.
        """
        print(self.description)
        print(self.usage)

    def execute(self) -> None:
        """
        Execute the command. This method should be overridden by each subclass.
        """
        raise NotImplementedError