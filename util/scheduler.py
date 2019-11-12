from torch.optim.lr_scheduler import _LRScheduler

class inv_lr_scheduler(_LRScheduler):
    def __init__(self, optimizer, alpha, beta, total_epoch, last_epoch=-1):
        self.alpha = alpha
        self.beta = beta
        self.total_epoch = total_epoch
        super(inv_lr_scheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [base_lr * ((1 + self.alpha * self.last_epoch / self.total_epoch) ** (-self.beta)) for base_lr in self.base_lrs]
