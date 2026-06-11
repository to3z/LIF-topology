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

class LIF_ring_neuron(nn.Module):
    """
    4 LIFs in a cross-timestep ring (0 -> 1 -> 2 -> 3 -> 0, with 1-step delay).
    Each LIF has its own m->n external projection. LIF_k receives its own
    external input plus the previous step's mem from LIF_{(k-1) mod 4}.
    Returns 4*channel features per timestep.
    """
    def __init__(self, in_planes, out_planes):
        super(LIF_ring_neuron, self).__init__()

        self.fc_0 = nn.Linear(in_planes, out_planes)
        self.fc_1 = nn.Linear(in_planes, out_planes)
        self.fc_2 = nn.Linear(in_planes, out_planes)
        self.fc_3 = nn.Linear(in_planes, out_planes)
        self.channel = out_planes
        self.thresh = thresh
        self.apply(weights_init_)

    def forward(self, input, wins=15):
        input = input.float().to(device)
        batch_size = input.size(0)
        mem = torch.zeros([batch_size, self.channel, 4], device=device)
        spike = torch.zeros([batch_size, self.channel, 4], device=device)
        spikes = torch.zeros([batch_size, wins, self.channel, 4], device=device)
        for step in range(wins):
            in_0 = self.fc_0(input) + mem[:, :, 3]
            in_1 = self.fc_1(input) + mem[:, :, 0]
            in_2 = self.fc_2(input) + mem[:, :, 1]
            in_3 = self.fc_3(input) + mem[:, :, 2]

            new_mem = torch.zeros_like(mem)
            new_spike = torch.zeros_like(spike)
            new_mem[:, :, 0], new_spike[:, :, 0] = mem_update(in_0, mem[:, :, 0], spike[:, :, 0])
            new_mem[:, :, 1], new_spike[:, :, 1] = mem_update(in_1, mem[:, :, 1], spike[:, :, 1])
            new_mem[:, :, 2], new_spike[:, :, 2] = mem_update(in_2, mem[:, :, 2], spike[:, :, 2])
            new_mem[:, :, 3], new_spike[:, :, 3] = mem_update(in_3, mem[:, :, 3], spike[:, :, 3])
            mem = new_mem
            spike = new_spike
            spikes[:, step, ...] = spike
        spikes = spikes.view(batch_size, wins, -1)
        return spikes
