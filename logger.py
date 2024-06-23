import sys
import os
import logging

def get_logger(log_dir, expt_name):
    logger = logging.getLogger("NeuralClothSim")
    logger.setLevel(logging.DEBUG)

    # Create handlers for logging to the standard output and a file
    stdoutHandler = logging.StreamHandler(stream=sys.stdout)    
    errHandler = logging.FileHandler(os.path.join(log_dir, f'{expt_name}.log'))

    # Set the log levels on the handlers
    stdoutHandler.setLevel(logging.DEBUG)
    errHandler.setLevel(logging.DEBUG)

    # Create a log format using Log Record attributes
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")

    # Set the log format on each handler
    stdoutHandler.setFormatter(fmt)
    errHandler.setFormatter(fmt)

    # Add each handler to the Logger object
    logger.addHandler(stdoutHandler)
    logger.addHandler(errHandler)
    return logger