import torch
import time
from cotracker.predictor import CoTrackerPredictor

device = 'cuda'
print("Loading model...")
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

B, T, C, H, W = 1, 8, 3, 384, 512
video = torch.rand((B, T, C, H, W), device=device)
queries = torch.rand((B, 100, 3), device=device)
queries[:, :, 0] = 0 # query from frame 0
queries[:, :, 1] *= W
queries[:, :, 2] *= H

print("Warming up...")
for _ in range(2):
    model(video, queries=queries)

print("Benchmarking...")
t0 = time.time()
n_iters = 10
for _ in range(n_iters):
    tracks, vis = model(video, queries=queries)
t1 = time.time()
print(f"Time per 8-frame chunk: {(t1-t0)/n_iters:.3f} s  ({n_iters / (t1-t0)} FPS chunks)")
