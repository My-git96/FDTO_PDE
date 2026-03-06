"""
学习率调度器模块
来源：/mnt/lv/litianyu/mycode/Gen-FVGN-3D/src/Utils/scheduler.py
"""
import math
from torch.optim.lr_scheduler import _LRScheduler


class ProgressiveRestartCosineAnnealingLR:
    """
    渐进式重启余弦退火学习率调度器

    每次重启时，最大学习率都会按指定因子递减，避免后期无意义震荡。

    用法示例：
    ```python
    scheduler = ProgressiveRestartCosineAnnealingLR(
        optimizer=optimizer,
        window_size=2000,           # 重启窗口大小
        total_windows=20,           # 重启窗口数量
        initial_max_lr=1e-3,        # 初始最大学习率
        decay_factor=0.5,           # 每次重启学习率衰减因子
        min_restart_lr=1e-4,        # 最低重启学习率
        eta_min=5e-5                # 每个周期内的最小学习率
    )
    ```
    """

    def __init__(self, optimizer, window_size, total_windows, initial_max_lr,
                 decay_factor, min_restart_lr, eta_min):
        self.optimizer = optimizer
        self.window_size = window_size
        self.total_windows = total_windows
        self.initial_max_lr = initial_max_lr
        self.decay_factor = decay_factor
        self.min_restart_lr = min_restart_lr
        self.eta_min = eta_min

        self.current_step = 0
        self.current_window = 0

        # 设置初始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_max_lr

    def get_current_max_lr(self):
        """计算当前窗口的最大学习率"""
        current_max = self.initial_max_lr * (self.decay_factor ** self.current_window)
        return max(current_max, self.min_restart_lr)

    def step(self):
        """更新学习率"""
        step_in_window = self.current_step % self.window_size

        # 检查是否需要重启到新窗口
        if step_in_window == 0 and self.current_step > 0:
            self.current_window += 1

        # 计算当前窗口的最大学习率
        current_max_lr = self.get_current_max_lr()

        # 在当前窗口内使用余弦退火
        progress = step_in_window / self.window_size
        lr = self.eta_min + (current_max_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2

        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1

    def get_last_lr(self):
        """获取最后一次的学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def reset(self):
        """重置调度器状态（用于新时间步）"""
        self.current_step = 0
        self.current_window = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_max_lr


class CosineAnnealingLR:
    """
    简单的余弦退火调度器（无重启）

    用法：
    ```python
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=1000,         # 总步数
        eta_min=1e-6        # 最小学习率
    )
    ```
    """

    def __init__(self, optimizer, T_max, eta_min=1e-6):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.current_step = 0

        # 记录初始学习率
        self.initial_lr = optimizer.param_groups[0]['lr']

    def step(self):
        """更新学习率"""
        # 余弦退火公式
        progress = min(self.current_step / self.T_max, 1.0)
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1

    def get_last_lr(self):
        """获取最后一次的学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def reset(self):
        """重置调度器状态"""
        self.current_step = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
