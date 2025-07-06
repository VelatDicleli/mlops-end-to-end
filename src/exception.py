import logging
import sys
from functools import wraps

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: sys) -> str:
        exc_type, exc_value, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "Unknown"
            line_number = -1

        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{error_message}]"

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"{self.__class__.__name__}({self.error_message})"


def exception_handler_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            custom_exc = CustomException(str(e), sys)
            logging.error(custom_exc)
            raise  
    return wrapper


