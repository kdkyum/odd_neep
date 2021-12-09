import torch
import torch.nn as nn

__all__ = [
    "mlp",
]


class NEEP1(nn.Module):
    def __init__(self, opt):
        super(NEEP1, self).__init__()
        self.n_layer = opt.n_layer
        self.h = nn.Sequential()
        self.h.add_module(
            "layer0",
            nn.Sequential(
                nn.Linear(2 * opt.N, opt.n_hidden),
                nn.ReLU(inplace=True),
            ),
        )

        for i in range(1, opt.n_layer):
            self.h.add_module(
                "layer%d" % i,
                nn.Sequential(
                    nn.Linear(opt.n_hidden, opt.n_hidden),
                    nn.ReLU(inplace=True),
                ),
            )
        self.h.add_module("out", nn.Linear(opt.n_hidden, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, x, v):
        s = torch.cat([x, v], dim=-1)
        _s = torch.cat([x, -v], dim=-1)
        return self.h(s) - self.h(_s)


class NEEP2(nn.Module):
    def __init__(self, opt):
        super(NEEP2, self).__init__()
        self.n_layer = opt.n_layer
        self.h = nn.Sequential()
        self.h.add_module(
            "layer0",
            nn.Sequential(
                nn.Linear(2 * opt.seq_len * opt.N, opt.n_hidden),
                nn.ReLU(inplace=True),
            ),
        )

        for i in range(1, opt.n_layer):
            self.h.add_module(
                "layer%d" % i,
                nn.Sequential(
                    nn.Linear(opt.n_hidden, opt.n_hidden),
                    nn.ReLU(inplace=True),
                ),
            )
        self.h.add_module("out", nn.Linear(opt.n_hidden, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, xs, vs):
        bsz = xs.size(1)
        s = torch.cat([xs, vs], dim=-1).transpose(0, 1)
        x = s.reshape(bsz, -1)

        _xs = torch.flip(xs, [0])
        _vs = torch.flip(-vs, [0])  # odd-parity
        _s = torch.cat([_xs, _vs], dim=-1).transpose(0, 1)
        _x = _s.reshape(bsz, -1)

        return self.h(x) - self.h(_x)


def mlp(opt):
    return NEEP1(opt).to(opt.device), NEEP2(opt).to(opt.device)
