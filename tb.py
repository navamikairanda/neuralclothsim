from torch.utils.tensorboard import SummaryWriter

def set_tensorboard_writer(log_dir):
    global writer
    writer = SummaryWriter(log_dir)