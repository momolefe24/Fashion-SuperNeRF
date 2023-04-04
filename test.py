import torch

from test_helper import *

render_poses = torch.Tensor(render_poses).to(device)
c2w = render_poses[0]
with torch.no_grad():
    rgb, disp, acc, _ = render(H, W, K, chunk=args.chunk, c2w=c2w[:3, :4], **render_kwargs_test)

print(rgb.shape)

