from test_helper import *

parser = config_parser()
args = parser.parse_args()

K = None

images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
i_train, i_val, i_test = i_split

near, far = 2., 6.

images = images[..., :3]
H, W, focal = hwf
H, W = int(H), int(W)
K = np.array([[focal, 0, 0.5*W],[0, focal, 0.5*H],[0, 0, 1]])


# Create NeRF network
render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
bds_dict = {
        'near' : near,
        'far' : far,
    }
render_kwargs_test.update(bds_dict)

