"""
Smoke test for topk switch in multi-task FC 5-LIF models.

Verifies:
1. topk=5 (default) and topk=4 both produce expected forward shapes
2. Backward gradients flow correctly in both modes
3. Gather 剔除 actually drops the slot with the lowest total spike count
"""
import sys
import os
import torch
import torch.nn as nn

MT_DIR = r"D:\MyProjects\neuron\complexity\multi-task"
sys.path.insert(0, MT_DIR)
os.chdir(MT_DIR)

# Monkey-patch device to cpu for the smoke test (the modules hardcode cuda:0).
import models.models_5LIF_fc.LIF_1_3_1_fc as m1
import models.models_5LIF_fc.LIF_1_1_3_fc as m2
import models.models_5LIF_fc.LIF_1_2_2_fc as m3
import models.models_5LIF_fc.LIF_1_1_2_1_fc as m4
import models.models_5LIF_fc.LIF_1_1_1_1_1_fc as m5
import models.models_5LIF_fc.LIF_1_4_fc as m6
for mod in (m1, m2, m3, m4, m5, m6):
    mod.device = torch.device("cpu")

MODULES = [
    ("LIF_1_3_1_fc", m1.SNN_Model_LIF_1_3_1),
    ("LIF_1_1_3_fc", m2.SNN_Model_LIF_1_1_3),
    ("LIF_1_2_2_fc", m3.SNN_Model_LIF_1_2_2),
    ("LIF_1_1_2_1_fc", m4.SNN_Model_LIF_1_1_2_1),
    ("LIF_1_1_1_1_1_fc", m5.SNN_Model_LIF_1_1_1_1_1),
    ("LIF_1_4_fc", m6.SNN_Model_LIF_1_4),
]


def smoke_test():
    torch.manual_seed(0)
    batch_size = 4
    n_tasks = 2
    n_classes = 10
    img_shape = (1, 36, 36)

    for name, Cls in MODULES:
        for topk in (5, 4):
            torch.manual_seed(0)  # reset for fair comparison
            model = Cls(n_tasks, topk=topk)
            # Confirm fc_output input dim
            expected_in = 512 * topk
            assert model.fc_output.in_features == expected_in, \
                f"{name} topk={topk}: fc_output.in_features={model.fc_output.in_features}, expected {expected_in}"
            # Forward
            x = torch.randn(batch_size, *img_shape)
            out = model(x, win=15)
            assert out.shape == (batch_size, n_tasks, n_classes), \
                f"{name} topk={topk}: out.shape={out.shape}, expected {(batch_size, n_tasks, n_classes)}"
            # Backward
            target = torch.randint(0, n_classes, (batch_size, n_tasks))
            loss = nn.functional.cross_entropy(out[:, 0, :], target[:, 0]) + \
                   nn.functional.cross_entropy(out[:, 1, :], target[:, 1])
            loss.backward()
            # Check fc_output got gradient
            assert model.fc_output.weight.grad is not None, \
                f"{name} topk={topk}: fc_output.weight.grad is None"
            assert model.fc_output.weight.grad.abs().sum() > 0, \
                f"{name} topk={topk}: fc_output.weight.grad is all zero"
            print(f"  [OK] {name} topk={topk}: fc_output in_features={model.fc_output.in_features}, "
                  f"out.shape={tuple(out.shape)}, grad flows")


def test_gather_drops_lowest():
    """Confirm that gather 剔除 actually drops the lowest-spike slot per sample."""
    torch.manual_seed(42)
    n_tasks = 2
    model = m1.SNN_Model_LIF_1_3_1(n_tasks, topk=4)
    # Forward and capture the post-gather h1_sumspike by monkey-patching the gather step.
    captured = {}

    # We monkey-patch Tensor.gather to log input (h1_sumspike.gather calls this method).
    def logging_gather(self, dim, index, **kwargs):
        result = self.gather.__wrapped__(self, dim, index, **kwargs) if hasattr(self.gather, '__wrapped__') else \
                 type(self).gather_orig(self, dim, index, **kwargs)
        if self.dim() == 3 and self.size(-1) == 5 and index.size(-1) == 4:
            captured['pre'] = self.detach().clone()
            captured['idx'] = index.detach().clone()
        return result

    # Simpler approach: temporarily wrap by replacing the method on the class.
    Tensor_cls = torch.Tensor
    Tensor_cls.gather_orig = Tensor_cls.gather
    Tensor_cls.gather = logging_gather
    try:
        x = torch.randn(3, 1, 36, 36)
        _ = model(x, win=15)
    finally:
        Tensor_cls.gather = Tensor_cls.gather_orig
        delattr(Tensor_cls, 'gather_orig')

    pre = captured['pre']  # [3, 512, 5]
    # Spike counts = pre.sum over channels
    counts = pre.sum(dim=1)  # [3, 5]
    # The code uses argsort(descending=True) and takes the first topk (=4).
    # So the "dropped" slot is whichever one argsort put last.
    # Equivalent to: dropped = sorted_idx[:, -1]
    sorted_idx = torch.argsort(counts, dim=1, descending=True)
    expected_dropped = sorted_idx[:, -1]  # [3]
    # The gather idx tells us which 4 of 5 slots are kept. The dropped slot
    # should NOT be in any kept set.
    idx = captured['idx']  # [3, 512, 4]
    # Check each sample: dropped slot must not appear in any of the 4 idx positions.
    for s in range(3):
        dropped = expected_dropped[s].item()
        kept = idx[s, 0, :].tolist()  # all channel indices are the same (gather was broadcast)
        assert dropped not in kept, \
            f"sample {s}: dropped slot {dropped} should NOT be in kept {kept}"
    print(f"  [OK] gather 剔除 drops the lowest-spike slot per sample "
          f"(dropped indices: {expected_dropped.tolist()})")


if __name__ == "__main__":
    print("Smoke test: topk=5 and topk=4 forward + backward (multi-task FC 5-LIF)")
    smoke_test()
    print()
    print("Smoke test: gather drops the lowest-spike slot per sample")
    test_gather_drops_lowest()
    print()
    print("All tests passed.")
