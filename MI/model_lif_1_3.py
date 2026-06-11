import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.integrate import odeint
from torch.distributions import Normal

v_min, v_max = -1e3, 1e3
thresh = 0.8
lens = 0.4
decay = 0.2
device = torch.device("cuda:0")

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

def mem_update(x, mem, spike):
    mem1 = mem * decay * (1. - spike) + x
    spike1 = act_fun(mem1)
    return mem1, spike1

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class LIF_1_3_neuron(nn.Module):
    """
    1+3 LIF topology (mem-mixing, LIF_hh style).

    Slot layout: [batch, channel, 4]
        slot 0 -> primary LIF (processes input)
        slots 1, 2, 3 -> mixing LIFs, each receives a learned non-negative
                          scalar of mem[:,:,0:1] via lif_fc = nn.Linear(1, 3)

    Mirrors LIF_hh's "1 residual follower" design but for a 1+3 layout
    (the 1 primary here is at slot 0, opposite to LIF_hh where the
    3 primaries are at slots 0,1,2 and the follower is at slot 3).
    """
    def __init__(self, in_planes, out_planes):
        super(LIF_1_3_neuron, self).__init__()
        # 1 primary LIF
        self.fc1 = nn.Linear(in_planes, out_planes)
        # 1 mixing layer producing 3 input currents (one per mixing LIF)
        self.lif_fc = nn.Linear(1, 3).to(device)
        self.lif_fc.weight.data = abs(self.lif_fc.weight.data)
        self.channel = out_planes
        self.thresh = thresh
        self.apply(weights_init_)

    def update_neuron(self, input, mem, spike):
        input_all = torch.zeros_like(mem)
        # Stage 1: primary, sees input
        input_all[:, :, 0] = self.fc1(input)
        # Stage 2: 3 mixing LIFs, see OLD mem[:,:,0] via lif_fc
        inner = self.lif_fc(mem[:, :, 0:1])
        input_all[:, :, 1] = inner[:, :, 0]
        input_all[:, :, 2] = inner[:, :, 1]
        input_all[:, :, 3] = inner[:, :, 2]
        mem1, spike1 = mem_update(input_all, mem, spike)
        return mem1, spike1

    def forward(self, input, wins=15):
        input = input.float().to(device)
        batch_size = input.size(0)
        mem = torch.zeros([batch_size, self.channel, 4], device=device)
        spike = torch.zeros([batch_size, self.channel, 4], device=device)
        spikes = torch.zeros([batch_size, wins, self.channel, 4], device=device)
        for step in range(wins):
            mem, spike = self.update_neuron(input, mem, spike)
            spikes[:, step, ...] = spike
        spikes = spikes.view(batch_size, wins, -1)
        return spikes
