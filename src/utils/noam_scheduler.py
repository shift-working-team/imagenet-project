class NoamScheduler():
    def __init__(self, optimizer, d_model, warmup_step, lr_scale=1):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_step = warmup_step
        self.lr_scale = lr_scale # 0.5가 적절할듯
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lrate = self.lr_scale * (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_step ** -1.5)
        self.optimizer.param_groups[0]['lr'] = lrate