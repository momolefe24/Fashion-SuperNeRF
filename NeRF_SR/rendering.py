import torch
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else "cpu"
'''
input: parsed json data(json_data), path to the dataset (dataset_path)
output: path to the images and corresponding camera-to-world matrices

Examples:
json_train_file: JSON File
ship_path: 'nerf_synthetic/ship'
ship_images = get_image_c2w(json_train_file,ship_path)[0] # The path to all the images

'''
def get_image_c2w(json_data,dataset_path):
    image_paths = [] # define a list to store the image paths
    c2ws = [] # define a list to store the camera2world matrices
    for frame in json_data['frames']:
        # grab the image file name
        image_path = frame['file_path']
        image_path = image_path.replace(".",dataset_path)
        image_paths.append(f"{image_path}.png")

        # grab the camera 2 world matrix
        c2ws.append(frame['transform_matrix'])
    # return the image file names
    return image_paths,c2ws

def get_meshgrid(width,height):
    x = np.linspace(0,width-1,width)
    y = np.linspace(0,height-1,height)
    return np.meshgrid(x,y)

def get_camera_vector(width,height,zc=0.5,focal_length=0.5):
    x,y = get_meshgrid(width,height)
    x_camera = (x-width*zc)/focal_length
    y_camera = (y-height*zc)/focal_length
    ones_like = np.ones(x_camera.shape)
    camera_vector = np.stack([x_camera,-y_camera,-ones_like],axis=-1)
    return camera_vector

def extend_dimension(camera_vector):
    return camera_vector[...,None,:]

def get_rays(width,height,camera2world,focal_length=0.5):
    camera_vector = get_camera_vector(width,height,focal_length=focal_length)
    camera_vector = extend_dimension(camera_vector)
    rotation = camera2world[:3,:3]
    translation = camera2world[:3,-1]
    # Get the world coordinates
    world_coordinates = camera_vector * rotation # (100,100,1,3) * (3,3) => (100,100,3,3)
#     print("World coordinates: ",world_coordinates.shape)
    # Calculate direction vector of the ray
    ray_d = np.sum(world_coordinates,axis=-1)  # (100,100,3)
#     print("RayD coordinates: ",ray_d.shape)
    ray_d = ray_d/np.linalg.norm(ray_d,axis=-1,keepdims=True)


    # Get origin vector of the ray
    ray_o = np.broadcast_to(translation,ray_d.shape)# (100,100,3)
#     print("RayO coordinates: ",ray_o.shape)
    return ray_o,ray_d

POS_ENCODE_DIMS = 16 #
def encode_position(x):
    """
    Encodes the position into its corresponding Fourier feature.
    Args:
        x: The input coordinate
    Returns:
        Fourier features tensors of the positions

    Shape:  POS_ENCODE_DIMS * 2[sin(x), cos(x)] * 3[Shape] + x[Original rays]
        Example: POS_ENCODE_DIMS = 16 * 2 * 3 = 96 + 3 = 99
    """
    positions = [x]
    for i in range(POS_ENCODE_DIMS):
        for fn in [np.sin,np.cos]:
            positions.append(fn((2.0 ** i) * x))
    return np.concatenate(positions,axis=-1)

def render_flat_rays(ray_o,ray_d,near,far,nC,rand=True): #nC is number of samples
    # Get the sample points from the ray, where each are 3D points (x,y,z0
    t_vals = np.linspace(near,far,nC)
    if rand:
        noise_shape = list(ray_o.shape[:-1]) + [nC]
        noise = (np.random.uniform(size=noise_shape)* (far - near)/nC)
        t_vals = t_vals + noise

    # Equation: r(t) = o + td -> Building the "r" here
    rays = ray_o[...,None,:] + (ray_d[...,None,:] * t_vals[...,None]) # parametric equation
    rays_flat = np.reshape(rays,[-1,3]) # (10000,3)
    rays_flat = encode_position(rays_flat)
    return (rays_flat,t_vals)


def map_fn(pose,focal_length,image_width,image_height,near,far,nC,rand=True):
    """
    Maps individual pose to flattened rays and sample points.

    Args:
        pose: The pose matrix of the camera.

    Returns:
        tuple of flattened rays and sample points corresponding to the camera pose
    """
    ray_o,ray_d = get_rays(image_width,image_height,pose,focal_length=focal_length)
    rays_flat,t_vals = render_flat_rays(ray_o,ray_d,near,far,nC,rand=rand)
    return rays_flat,t_vals

trans_t = lambda t: tf.convert_to_tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1],
], dtype=tf.float32)

rot_phi = lambda phi: tf.convert_to_tensor([
    [1, 0, 0, 0],
    [0, tf.cos(phi), -tf.sin(phi), 0],
    [0, tf.sin(phi), tf.cos(phi), 0],
    [0, 0, 0, 1],
], dtype=tf.float32)

rot_theta = lambda th: tf.convert_to_tensor([
    [tf.cos(th), 0, -tf.sin(th), 0],
    [0, 1, 0, 0],
    [tf.sin(th), 0, tf.cos(th), 0],
    [0, 0, 0, 1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w