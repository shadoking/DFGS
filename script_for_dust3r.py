import sys
sys.path.append("DUSt3R")
import os
from os import makedirs
import numpy as np
from PIL import Image as PILImage
from PIL.ImageOps import exif_transpose
from argparse import ArgumentParser

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

from scipy.spatial.transform import Rotation
import torchvision.transforms as tvf

from scene.dataset_readers import storePly
from utils.read_write_model import (
    Camera, BaseImage, Point3D, 
    rotmat2qvec, 
    write_cameras_binary, 
    write_images_binary,
    write_points3D_binary, 
    write_cameras_text, 
    write_images_text)

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PILImage.LANCZOS
    elif S <= long_edge_size:
        interp = PILImage.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PILImage.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs, (W1, H1)

def save_cameras(sparse_path, rsz_focals, org_img_shape, rsz_img_shape):
    output_camera_bin_path = os.path.join(sparse_path,  "cameras.bin")
    output_camera_txt_path = os.path.join(sparse_path,  "cameras.txt")
    
    original_w, original_h = org_img_shape
    resized_h, resized_w = rsz_img_shape[1:3]
    
    scale_w = original_w / resized_w
    scale_h = original_h / resized_h
    
    cameras = {}
    for i, focal in enumerate(rsz_focals, start=1):
        cameras[i] = Camera(
            id=i,
            model="PINHOLE",
            width=original_w,
            height=original_h,
            params=[focal*scale_w, focal*scale_h, original_w/2, original_h/2]
        )    

    write_cameras_binary(cameras, output_camera_bin_path)
    write_cameras_text(cameras, output_camera_txt_path)

def save_iamges(sparse_path, c2ws, image_names):
    output_image_bin_path = os.path.join(sparse_path,  "images.bin")
    output_image_txt_path = os.path.join(sparse_path,  "images.txt")
    
    w2cs = np.linalg.inv(c2ws)  
    images = {}
    for i, (w2c, image_name) in enumerate(zip(w2cs, image_names), start=1):
        qvec = rotmat2qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        images[i] = BaseImage(
            id=i,
            qvec=qvec,
            tvec=tvec,
            camera_id=i,
            name=image_name,
            xys=[],
            point3D_ids=[])
     
    write_images_binary(images, output_image_bin_path)
    write_images_text(images, output_image_txt_path)

def save_points(sparse_path, ori_points, images, mask, batch_indices='all', use_mask=True):
    output_point_path = os.path.join(sparse_path,  "points3D.ply")
    
    # 根据点数选择
    # points = np.concatenate([p[m] for p, m in zip(ori_points, mask)])
    # col = np.concatenate([p[m] for p, m in zip(images, mask)])
    # colors = (col * 255).astype(np.uint8)
    
    # if num_points != 'all' and num_points < points.shape[0]:
    #     indices = np.random.choice(points.shape[0], num_points, replace=False)
    #     points = points[indices]
    #     colors = colors[indices]
    
    ori_points = np.array(ori_points)
    images = np.array(images)
    mask = np.array(mask)
    
    if batch_indices == 'all':
        selected_points = ori_points
        selected_images = images
        selected_mask = mask
    else:
        selected_points = ori_points[batch_indices]
        selected_images = images[batch_indices]
        selected_mask = mask[batch_indices]
    
    if use_mask:
        points = np.concatenate([p[m] for p, m in zip(selected_points, selected_mask)])
        col = np.concatenate([p[m] for p, m in zip(selected_images, selected_mask)])
    else:
        points = selected_points.reshape(-1, 3)
        col = selected_images.reshape(-1, 3)
        
    colors = (col * 255).astype(np.uint8)
    
    storePly(output_point_path, points, colors)
    
def run_dust3r(image_paths, model_name=None, device="cuda", batch_size=1, schedule='cosine', lr=0.01, niter=300, silent=True):
    if model_name is None:
        model_name = "DUSt3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    images, ori_image_shape = load_images(image_paths, size=512, verbose=not silent)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    mode = GlobalAlignerMode.PointCloudOptimizer if len(images) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
    
    images = np.array(scene.imgs)
    pts3d = to_numpy(scene.get_pts3d())
    mask = to_numpy(scene.get_masks())
    
    focals = to_numpy(scene.get_focals())
    poses = to_numpy(scene.get_im_poses())
    
    return images, pts3d, mask, focals, poses, ori_image_shape

if __name__ == '__main__':
    parser = ArgumentParser(description="Run DUSt3R")
    parser.add_argument("--source_path", "-s", type=str)
    
    args = parser.parse_args(sys.argv[1:])
    
    image_dir = os.path.join(args.source_path, "images")
    if not os.path.exists(image_dir):
        print(f"Error: The folder does not exist: {image_dir}")
        sys.exit(1)
    
    image_paths = []
    image_names = []
    for item in os.listdir(image_dir):
        image_names.append(item)
        image_paths.append(os.path.join(image_dir, item))
    
    if len(image_paths) == 0:
        print(f"Error: The images do not exist!")
        sys.exit(1)
    
    rsz_images, ori_points, mask, rsz_focals, c2ws, ori_image_shape = run_dust3r(image_paths)

    sparse_path = os.path.join(args.source_path, "sparse", "0")
    makedirs(sparse_path, exist_ok=True)
    
    # cameras
    save_cameras(sparse_path, rsz_focals, ori_image_shape, rsz_images.shape)
     
    # iamges
    save_iamges(sparse_path, c2ws, image_names)
    
    # points
    save_points(sparse_path, ori_points, rsz_images, mask)
