from torch.optim.lr_scheduler import LambdaLR

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps):
        """
        创建一个支持 Warm Up 的学习率调度器，继承自 LambdaLR

        Args:
            optimizer (Optimizer): PyTorch 优化器实例
            warmup_steps (int): Warm Up 的步数
            base_lr (float): 最终的目标学习率
            warmup_lr (float): Warm Up 时的初始学习率
        """
        self.warmup_steps = warmup_steps
        
        # LambdaLR 的学习率更新函数
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                return 1.0  # 完成 warmup 后使用 base_lr
        
        # 使用 LambdaLR 实例
        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    def state_dict(self):
        """获取当前学习率调度器的状态"""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载学习率调度器的状态"""
        self.scheduler.load_state_dict(state_dict)

    def step(self):
        """更新学习率"""
        self.scheduler.step()

    def get_last_lr(self):
        """获取当前学习率"""
        return self.scheduler.get_last_lr()
