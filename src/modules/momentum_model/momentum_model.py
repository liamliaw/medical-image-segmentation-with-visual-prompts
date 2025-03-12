import torch.nn as nn


class MomentumModel(nn.Module):
    def __init__(self, conf, architecture):
        super().__init__()
        self.tau = conf.tau
        self.net_student = architecture(conf=conf)
        self.net_teacher = architecture(conf=conf)

    def copy_state_dict(self):
        for (_, param_student), (_, param_teacher) in zip(
            self.net_student.named_parameters(),
            self.net_teacher.named_parameters(),
        ):
            param_teacher.data.copy_(param_student.data)
            param_teacher.requires_grad = False

    def forward(self, x_students, x_teacher):
        out_sts = []
        for x_st in x_students:
            out_st = self.net_student(x_st)
            out_sts.append(out_st)
        out_tch = self.net_teacher(x_teacher)
        return out_sts, out_tch

    def update_teacher(self):
        for (_, param_student), (_, param_teacher) in zip(
            self.net_student.named_parameters(),
            self.net_teacher.named_parameters(),
        ):
            # EMA update.
            param_teacher.data = (
                self.tau * param_teacher.data
                + (1 - self.tau) * param_student.data
            )