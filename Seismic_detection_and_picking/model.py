import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation="relu", output_dim=1):
        super(MLP, self).__init__()

        layers = []

        # 选择激活函数
        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "leakyrelu":
            act_fn = nn.LeakyReLU()
        elif activation.lower() == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError("Unsupported activation")

        # 第一层
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())  # 二分类必须

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)