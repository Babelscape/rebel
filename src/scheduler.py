
from torch.optim.lr_scheduler import LambdaLR

def get_inverse_square_root_schedule_with_warmup(
    optimizer, num_warmup_steps, warmup_init_lr=-1, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by `lr_end`, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        lr (:obj:`float`, `optional`, defaults to 1e-7):
            The  LR.
        warmup_init_lr ():
            The initial lr. Defaults to LR.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """
    warmup_end_lr = optimizer.defaults["lr"]
    if warmup_init_lr < 0:
        warmup_init_lr = 0 if num_warmup_steps > 0 else warmup_end_lr

    # linearly warmup for the first args.warmup_updates
    lr_step = (warmup_end_lr - warmup_init_lr) / num_warmup_steps

    # then, decay prop. to the inverse square root of the update number
    decay_factor =  num_warmup_steps ** 0.5

    # initial learning rate
    lr = warmup_init_lr

    # optimizer.set_lr(lr)

    def lr_lambda(current_step: int):
        """Update the learning rate after each update."""
        if current_step < num_warmup_steps:
            lr = warmup_init_lr + current_step * lr_step
            lr = lr/warmup_end_lr
        else:
            lr = decay_factor * current_step ** -0.5
        return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)