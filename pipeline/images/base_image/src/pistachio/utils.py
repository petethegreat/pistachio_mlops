# utils

import os
import sys

def ensure_directory_exists(path_to_check: str) ->None:
    """check directory for path exists - if not, create it

    Args:
        path (str): intended file path
    """

    the_dir = os.path.dirname(path_to_check)
    if not os.path.exists(the_dir):
        os.makedirs(the_dir)
#####################################################################