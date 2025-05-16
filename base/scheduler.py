import math

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler


__all__ = ['GradualWarmupScheduler', 'MyWarmupScheduler', 'MyStepLR',
           'MyCosineLR']


class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up
         with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg.
        ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr  for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr for base_lr in self.base_lrs]

        return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        self.last_epoch = 1
        if epoch is not None:
            self.last_epoch = epoch + 2

        if self.last_epoch < self.total_epoch:
            warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        elif self.last_epoch == self.total_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr']
        else:
            # If not using warmup, then do nothing at the beginning.
            if self.last_epoch == 1 and self.total_epoch == 0:
                pass
            else:
                self.after_scheduler.step(metrics)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class MyWarmupScheduler(object):
    def __init__(self, optimizer, lr, min_lr, best=None, mode='min',
                 patience=5, factor=0.1, num_warmup_epoch=20, init_epoch=0,
                 eps=1e-11, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.learning_rate = lr

        self.best = best
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.num_warmup_epoch = num_warmup_epoch
        self.init_epoch = init_epoch
        self.relative_epoch = 0
        self.last_epoch = -1
        self.last_stage_epoch = None
        self.num_bad_epochs = 0
        self.eps = eps
        self.verbose = verbose

        if best is None:
            self.best = -1e10
            if mode == "min":
                self.best = 1e10

    def is_better(self, metric):

        better = 0
        if self.mode == "min":
            if metric < self.best:
                better = 1
        else:
            if metric > self.best:
                better = 1

        return better

    def warmup_lr(self, init_lr, batch,  num_batch_warm_up):
        if self.relative_epoch < self.num_warmup_epoch:
            for params in self.optimizer.param_groups:
                params['lr'] = batch * init_lr * (self.relative_epoch + 1) / (num_batch_warm_up * self.num_warmup_epoch + 1e-100)


    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            # new_lr = max(old_lr * self.factor, self.min_lrs[i])
            new_lr = old_lr * self.factor
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    def step(self, epoch, metrics):
        self.relative_epoch = epoch - self.init_epoch + 1

        if self.relative_epoch == self.num_warmup_epoch:
            for params in self.optimizer.param_groups:
                params['lr'] = self.learning_rate

        current = float(metrics)

        if self.is_better(current):
            self.best = current
            self.num_bad_epochs = 0
        else:
            if self.relative_epoch > self.num_warmup_epoch:
                self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.num_bad_epochs = 0


class MyStepLR(lr_scheduler.StepLR):
    """
    Override: https://pytorch.org/docs/1.0.0/_modules/
    torch/optim/lr_scheduler.html#StepLR
    Reason: we want to fix the learning rate to not get lower than some value:
    min_lr.

    Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        min_lr (float): The lowest allowed value for the learning rate.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1,
                 min_lr=1e-6):
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super(lr_scheduler.StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** (self.last_epoch // self.step_size),
                self.min_lr) for base_lr in self.base_lrs]


class MyCosineLR(lr_scheduler.StepLR):
    """
    Override: https://pytorch.org/docs/1.0.0/_modules/torch/optim/
    lr_scheduler.html#StepLR
    Reason: use a cosine evolution of lr.
    paper:
    `S. Qiao, W. Shen, Z. Zhang, B. Wang, and A. Yuille.  Deepco-training for
    semi-supervised image recognition. InECCV,2018`


    for the epoch T:
    lr = base_lr * coef × (1.0 + cos((T − 1) × π/max_epochs)).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        coef (float): float coefficient. e.g. 0.005
        max_epochs (int): maximum epochs.
        last_epoch (int): The index of last epoch. Default: -1.
        min_lr (float): The lowest allowed value for the learning rate.
    """

    def __init__(self, optimizer, coef, max_epochs, min_lr=1e-9,
                 last_epoch=-1):
        assert isinstance(coef, float), "'coef' must be a float. found {}" \
                                        "...[not ok]".format(type(coef))
        assert coef > 0., "'coef' must be > 0. found {} ....[NOT OK]".format(
            coef
        )
        assert max_epochs > 0, "'max_epochs' must be > 0. found {}" \
                               "...[NOT OK]".format(max_epochs)
        self.max_epochs = float(max_epochs)
        self.coef = coef
        self.min_lr = min_lr
        super(lr_scheduler.StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.coef * (
                    1. + math.cos((self.last_epoch - 1) * math.pi /
                                  self.max_epochs)),
                self.min_lr) for base_lr in self.base_lrs]


if __name__ == "__main__":
    from torch.optim import SGD
    import torch
    import matplotlib.pyplot as plt

    optimizer = SGD(torch.nn.Linear(10, 20).parameters(), lr=0.001)
    lr_sch = MyCosineLR(optimizer, coef=0.5, max_epochs=600, min_lr=1e-9)
    vals = []
    for i in range(1000):
        optimizer.step()
        vals.append(lr_sch.get_lr())
        lr_sch.step()
    plt.plot(vals)
    plt.savefig("lr_evolve_{}.png".format(lr_sch.__class__.__name__))
