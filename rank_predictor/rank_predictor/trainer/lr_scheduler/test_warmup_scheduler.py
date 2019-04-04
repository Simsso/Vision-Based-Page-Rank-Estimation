from unittest import TestCase

import torch
from rank_predictor.trainer.lr_scheduler.warmup_scheduler import GradualWarmupScheduler


class TestWarmupScheduler(TestCase):

    def test_lr_development(self):
        """Just here for plotting purposes."""
        opt = torch.optim.Adam(torch.nn.Linear(10, 10).parameters(), lr=1)

        scheduler_plateau = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
        scheduler_warmup = GradualWarmupScheduler(opt, multiplier=8, total_epoch=10,
                                                  after_scheduler=scheduler_plateau)

        for epoch in range(50):
            scheduler_warmup.step()  # 10 epoch warmup, after that schedule as scheduler_plateau
            for param_group in opt.param_groups:
                #print(param_group['lr'])
                pass
