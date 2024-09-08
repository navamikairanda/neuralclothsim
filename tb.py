
def set_tensorboard_writer(log_dir):
    global writer
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except ImportError:
        writer = None