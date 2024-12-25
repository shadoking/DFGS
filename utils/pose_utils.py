import numpy as np

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.
    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
        A tuple (poses, transform), with the transformed poses and the applied
        camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform

def generate_ellipse_path(views):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)
    
    ts = poses[:, :3, 3]
    t_thetas = np.arctan2(ts[:, 1], ts[:, 0])

    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset

    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

     # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return transform, center, up, low, high, z_low, z_high, ts, t_thetas


def generate_random_poses_annealing_view(views, n_frames=10000):
    """Generates random poses."""
    init_poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        init_poses.append(tmp_view)
    init_poses = np.stack(init_poses, 0)
    poses, transform = transform_poses_pca(init_poses)

    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    
    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    
    poses_num = len(poses)
    t_std_start = 0.2 * n_frames
    t_std_max = 0.05
    z_std_start = 0.2 * n_frames
    z_std_max = 0.05
    
    random_poses = []
    closest_poses = []

    for idx in range(n_frames):
        
        pose_idx = np.random.randint(poses_num)
        selected_pose = poses[pose_idx]
        
        selected_position = selected_pose[:3, 3]
        selected_t = selected_position
        
        assert (selected_t < -1).sum() == 0 and (selected_t > 1).sum() == 0
        
        t_std = max(t_std_start, float(idx+1)) / n_frames * t_std_max
        t_noise = np.random.randn(3) * t_std
        noisy_t = selected_t + t_noise
        noisy_position = noisy_t
        noisy_position = np.clip(noisy_position, -1, 1)
        
        lookat = center
        z_std = max(z_std_start, float(idx+1)) / n_frames * z_std_max
        z_noise = np.random.randn(3) * z_std
        noisy_lookat = lookat + z_noise
        
        noisy_z = noisy_position - noisy_lookat
        
        random_pose = np.eye(4)
        random_pose[:3] = viewmatrix(noisy_z, up, noisy_position)
        random_pose = np.linalg.inv(transform) @ random_pose
        random_pose[:3, 1:3] *= -1
        random_pose = np.linalg.inv(random_pose)
        random_poses.append(random_pose)

        closest_poses.append(views[pose_idx])
    
    return random_poses, closest_poses