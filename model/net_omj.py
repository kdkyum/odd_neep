import torch
import torch.nn as nn

__all__ = [
    "mlp",
]


class NEEP1(nn.Module):
    def __init__(self, opt):
        super(NEEP1, self).__init__()
        self.N = opt.N
        self.n_layer = opt.n_layer
        self.encoder = nn.Embedding(2 * opt.N, opt.n_hidden // 2)
        self.h = nn.Sequential()
        self.h.add_module(
            "layer0",
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(opt.n_hidden // 2, opt.n_hidden),
                nn.ReLU(inplace=True),
            ),
        )

        for i in range(1, opt.n_layer + 1):
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

    def forward(self, x):
        x_f = self.encoder(x)
        x_r = (x + self.N) % (2 * self.N)
        x_r = self.encoder(x_r)
        return self.h(x_f) - self.h(x_r)


class NEEP2(nn.Module):
    def __init__(self, opt):
        super(NEEP2, self).__init__()
        self.n_layer = opt.n_layer
        self.N = opt.N
        self.h = nn.Sequential()
        self.encoder = nn.Embedding(2 * opt.N, opt.n_hidden // 2)
        self.h.add_module(
            "layer0",
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(2 * (opt.n_hidden // 2), opt.n_hidden),
                nn.ReLU(inplace=True),
            ),
        )

        for i in range(1, opt.n_layer + 1):
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

    def forward(self, xs):
        bsz = xs.size(0)
        xs_f = self.encoder(xs)
        xs_f = xs_f.reshape(bsz, -1)
        xs_r = (xs + self.N) % (2 * self.N)
        xs_r = self.encoder(xs_r)
        xs_r = torch.flip(xs_r, [1]).reshape(bsz, -1)
        return self.h(xs_f) - self.h(xs_r)


class NEEP_WTD(nn.Module):
    def __init__(self, opt):
        super(NEEP_WTD, self).__init__()
        self.n_layer = opt.n_layer
        self.N = opt.N
        self.encoder = nn.Embedding(2 * opt.N, opt.n_hidden // 2)
        self.h = nn.Sequential()
        self.h.add_module(
            "layer0",
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(opt.n_hidden // 2 + 1, opt.n_hidden),
                nn.ReLU(inplace=True),
            ),
        )

        for i in range(1, opt.n_layer + 1):
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

    def forward(self, x, t):
        x_f = self.encoder(x)
        t = t.unsqueeze(-1)
        s_f = torch.cat([x_f, t], dim=-1)
        x_r = (x + self.N) % (2 * self.N)
        x_r = self.encoder(x_r)
        s_r = torch.cat([x_r, t], dim=-1)
        return self.h(s_f) - self.h(s_r)


def mlp(opt):
    return (
        NEEP1(opt).to(opt.device),
        NEEP2(opt).to(opt.device),
        NEEP_WTD(opt).to(opt.device),
    )
