import sys
import os
import logging

def get_logger(log_dir, expt_name):
    logger = logging.getLogger("NeuralClothSim")
    logger.setLevel(logging.DEBUG)

    stdoutHandler = logging.StreamHandler(stream=sys.stdout)    
    errHandler = logging.FileHandler(os.path.join(log_dir, f'{expt_name}.log'))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
    stdoutHandler.setFormatter(fmt)
    errHandler.setFormatter(fmt)

    logger.addHandler(stdoutHandler)
    logger.addHandler(errHandler)
    return logger