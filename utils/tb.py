
def set_tensorboard_writer(log_dir, debug):
    global writer
    writer = None
    if debug:          
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir)
        except ImportError:
            pass