import argparse
import logging
from utils.command_handler import CommandHandler
from utils.command_parser import CommandParser

# TODO 1-1: Use argparse to parse the command line arguments (verbose and log_file).
# TODO 1-2: Set up logging and initialize the logger object.

def set_logger(log_path):
    logging.basicConfig(level=logging.INFO, 
                        filename=log_path, 
                        filemode="w",
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logger = logging.getLogger(__name__)

    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help='verbose mode', action='store_true') # --verbose 사용 시 true 저장
    parser.add_argument("--log_path", help='log path', default='file_explorer.log') 
    args = parser.parse_args()

    logger = set_logger(args.log_path)
    command_parser = CommandParser(args.verbose)
    handler = CommandHandler(command_parser)
    
    while True:
        command = input(">> ")
        handler.execute(command)
        logger.info(f"Input command: {command}")


if __name__=="__main__":
    main()