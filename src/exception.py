import os
import sys
from logger import log


def error_message_detail(error_message, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    message = f"Error occured in file-name: {file_name} | Line-no: {line_no} | Cause: {error_message}"

    return message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.message

