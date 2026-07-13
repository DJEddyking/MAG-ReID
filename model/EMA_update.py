
import torch
import copy


class EMA:
    def __init__(self, student_model, teacher_model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        # 初始化 teacher 为 student 的拷贝（外部也可直接传入已拷贝的 teacher）
        teacher_model.load_state_dict(student_model.state_dict())
        # 确保 teacher 不需要梯度
        for p in teacher_model.parameters():
            p.requires_grad = False
        self.teacher = teacher_model
        # 用 shadow dict 存储浮点副本，避免直接依赖 teacher param 的 grad 状态
        self.shadow = {}
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device) if device else param.data.clone()

    @torch.no_grad()
    def update(self, student_model):
        # 更新 shadow
        for name, param in student_model.named_parameters():
            if name not in self.shadow:
                continue
            new = param.data
            old = self.shadow[name]
            # 保持 device/dtype 一致
            if self.device:
                new = new.to(self.device)
            self.shadow[name] = old * self.decay + (1.0 - self.decay) * new

        # 把 shadow 写回 teacher 参数（teacher 不参与 grad）
        for name, param in self.teacher.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def apply_to_teacher(self, model):
        # 可选：把当前 shadow 强制写回任意模型（例如用于保存前同步）
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        for p in model.parameters():
            p.requires_grad = False