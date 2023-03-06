# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/modules/lr_schedulers.py
# reference: https://github.com/lifeiteng/vall-e
import math

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam, Optimizer
import contextlib
import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from torch.optim.lr_scheduler import ExponentialLR


class ExpLR:
    def __init__(self, optimizer, final_epoch, decay_factor):
        self.optimizer = optimizer
        self.final_epoch = final_epoch
        self.decay_factor = decay_factor
        self.gamma = self._calculate_gamma()
        self.scheduler = ExponentialLR(optimizer, gamma=self.gamma)

    def _calculate_gamma(self):
        return self.decay_factor ** (1 / self.final_epoch)

    def step_epoch(self):
        self.scheduler.step()

    def step_batch(self):
        pass  # This method intentionally does nothing
    
    def state_dict(self):
        """Returns the state of the scheduler as a dictionary."""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the scheduler's state."""
        self.scheduler.load_state_dict(state_dict)


class LRScheduler(object):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch.
    """

    def __init__(self, optimizer: Optimizer, verbose: bool = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]

        self.epoch = 0
        self.batch = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            "base_lrs": self.base_lrs,
            "epoch": self.epoch,
            "batch": self.batch,
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler.  Will be a list of float."""
        return self._last_lr

    def get_lr(self):
        # Compute list of learning rates from self.epoch and self.batch and
        # self.base_lrs; this must be overloaded by the user.
        # e.g. return [some_formula(self.batch, self.epoch, base_lr) for base_lr in self.base_lrs ]
        raise NotImplementedError

    def step_batch(self, batch: Optional[int] = None) -> None:
        # Step the batch index, or just set it.  If `batch` is specified, it
        # must be the batch index from the start of training, i.e. summed over
        # all epochs.
        # You can call this in any order; if you don't provide 'batch', it should
        # of course be called once per batch.
        if batch is not None:
            self.batch = batch
        else:
            self.batch = self.batch + 1
        self._set_lrs()

    def step_epoch(self, epoch: Optional[int] = None):
        # Step the epoch index, or just set it.  If you provide the 'epoch' arg,
        # you should call this at the start of the epoch; if you don't provide the 'epoch'
        # arg, you should call it at the end of the epoch.
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch = self.epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        assert len(values) == len(self.optimizer.param_groups)

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr):
        """Display the current learning rate."""
        if is_verbose:
            logging.warn(
                f"Epoch={self.epoch}, batch={self.batch}: adjusting learning rate"
                f" of group {group} to {lr:.4e}."
            )


class Eden(LRScheduler):
    """
    Eden scheduler.
    The basic formula (before warmup) is:
      lr = base_lr * (((batch**2 + lr_batches**2) / lr_batches**2) ** -0.25 *
                     (((epoch**2 + lr_epochs**2) / lr_epochs**2) ** -0.25)) * warmup
    where `warmup` increases from linearly 0.5 to 1 over `warmup_batches` batches
    and then stays constant at 1.

    If you don't have the concept of epochs, or one epoch takes a very long time,
    you can replace the notion of 'epoch' with some measure of the amount of data
    processed, e.g. hours of data or frames of data, with 'lr_epochs' being set to
    some measure representing "quite a lot of data": say, one fifth or one third
    of an entire training run, but it doesn't matter much.  You could also use
    Eden2 which has only the notion of batches.

    We suggest base_lr = 0.04 (passed to optimizer) if used with ScaledAdam

    Args:
        optimizer: the optimizer to change the learning rates on
        lr_batches: the number of batches after which we start significantly
              decreasing the learning rate, suggest 5000.
        lr_epochs: the number of epochs after which we start significantly
              decreasing the learning rate, suggest 6 if you plan to do e.g.
              20 to 40 epochs, but may need smaller number if dataset is huge
              and you will do few epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: int = 5000,
        lr_epochs: int = 10,
        warmup_batches: int = 500.0,
        warmup_start: float = 0.5,
        verbose: bool = False,
    ):
        super(Eden, self).__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs
        self.warmup_batches = warmup_batches

        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start

    def get_lr(self):
        factor = (
            (self.batch**2 + self.lr_batches**2) / self.lr_batches**2
        ) ** -0.25 * (
            ((self.epoch**2 + self.lr_epochs**2) / self.lr_epochs**2) ** -0.25
        )
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
            # else 0.5 + 0.5 * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]

class WarmupCosineLRSchedule(torch.optim.lr_scheduler._LRScheduler):
    """
    Implements Warmup learning rate schedule until 'warmup_steps', going from 'init_lr' to 'peak_lr' for multiple optimizers.
    """

    def __init__(
        self,
        optimizer,
        init_lr,
        peak_lr,
        end_lr,
        warmup_steps=10000,
        total_steps=400000,
        current_step=0,
    ):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.optimizer = optimizer
        self._warmup_rate = (peak_lr - init_lr) / warmup_steps
        self._decay_rate = (end_lr - peak_lr) / (total_steps - warmup_steps)
        self._current_step = current_step
        self.lr = init_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._last_lr = [self.lr]

    def set_lr(self, lr):
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def step(self):
        if self._current_step < self.warmup_steps:
            lr = self.init_lr + self._warmup_rate * self._current_step

        elif self._current_step > self.total_steps:
            lr = self.end_lr

        else:
            decay_ratio = (self._current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            if decay_ratio < 0.0 or decay_ratio > 1.0:
                raise RuntimeError(
                    "Decay ratio must be in [0.0, 1.0]. Fix LR scheduler settings."
                )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.end_lr + coeff * (self.peak_lr - self.end_lr)

        self.set_lr(lr)
        self.lr = lr
        self._current_step += 1
        return self.lr


if __name__ == "__main__":
    m = nn.Linear(10, 10)
    opt = Adam(m.parameters(), lr=1e-4)
    s = WarmupCosineLRSchedule(
        opt, 1e-6, 2e-4, 1e-6, warmup_steps=2000, total_steps=20000, current_step=0
    )
    lrs = []
    for i in range(25000):
        s.step()
        lrs.append(s.lr)
        print(s.lr)

    plt.plot(lrs)
    plt.plot(range(0, 25000), lrs)
    plt.show()