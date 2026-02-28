"""
Test script to verify the fake distributed implementation.
"""

import torch
from nandmachine.frontend.network import hook_dist

print("=" * 60)
print("Testing Fake Distributed Implementation")
print("=" * 60)

# Test 1: Basic API
print("\n[Test 1] Basic API")
print(f"Rank: {hook_dist.get_rank()}")  # Should print 0
print(f"World Size: {hook_dist.get_world_size()}")  # Should print 1

# Test 2: Change world size
print("\n[Test 2] Change world size")
hook_dist.set_world_size(4)
print(f"World Size after change: {hook_dist.get_world_size()}")  # Should print 4

# Test 3: all_reduce with regular tensor
print("\n[Test 3] all_reduce with regular tensor")
tensor = torch.randn(3, 4)
result = hook_dist.all_reduce(tensor)
print(f"Input tensor shape: {tensor.shape}")
print(f"All reduce result shape: {result.shape}")  # Should print torch.Size([3, 4])
print(f"Tensors are same: {torch.equal(tensor, result)}")  # Should be True

# Test 4: all_reduce with meta tensor
print("\n[Test 4] all_reduce with meta tensor")
meta_tensor = torch.empty(5, 6, device='meta')
meta_result = hook_dist.all_reduce(meta_tensor)
print(f"Meta tensor shape: {meta_tensor.shape}")
print(f"Meta tensor result shape: {meta_result.shape}")  # Should print torch.Size([5, 6])
print(f"Meta tensor device: {meta_result.device}")  # Should print meta

# Test 5: Import and use in torch_kernels
print("\n[Test 5] Import torch_kernels module")
try:
    from nandmachine.frontend.network.torch_kernels import LinearBase, ColumnParallelLinear
    print("[OK] Successfully imported torch_kernels classes")
except Exception as e:
    print(f"[FAIL] Failed to import: {e}")

# Test 6: Import and use Qwen3Attention
print("\n[Test 6] Import qwen3 module")
try:
    from nandmachine.frontend.network.qwen3 import Qwen3Attention
    print("[OK] Successfully imported Qwen3Attention")
except Exception as e:
    print(f"[FAIL] Failed to import: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
