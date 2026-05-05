import torch
import time

device = 'cuda'
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

B, T, C, H, W = 1, 2, 3, 384, 512
video = torch.rand((B, T, C, H, W), device=device)
queries = torch.rand((B, 100, 3), device=device)
queries[:, :, 0] = 0

for _ in range(2): model(video, queries=queries)
t0 = time.time()
n_iters = 20
for _ in range(n_iters):
    tracks, vis = model(video, queries=queries)
t1 = time.time()
print(f"Time per 2-frame chunk: {(t1-t0)/n_iters:.3f} s")
