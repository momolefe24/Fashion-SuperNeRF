# from test_helper import *
#
# rgbs = []
# for i in range(4):
# 	rgb = probe(torch.tensor(poses[i]), i)
# 	save_image(f"r_{str(i)}", rgb)
# 	print(f"Image r_{i} saved")
# 	rgbs.append(rgb)
#
# imageio.mimwrite('nerf.mp4', to8b(rgbs), fps=30, quality=8)
