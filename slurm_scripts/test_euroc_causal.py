import argparse
import glob
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import lietorch
import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "droid_slam"))

from depth_video import DepthVideo
from droid_frontend import DroidFrontend
from droid_net import DroidNet
from motion_filter import MotionFilter
from trajectory_filler import PoseTrajectoryFiller

# Besides using the EUROC timestamps all other changes to the original image_stream
# of test_euroc.py are cosmetic (lower case variable names, path library, ...)
def image_stream(datapath, image_size=(320, 512), stereo=False, stride=1):
    """EuRoC ASL image stream with the rectification used by test_euroc.py."""

    k_l = np.array(
        [458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]
    ).reshape(3, 3)
    d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
    r_l = np.array(
        [
            0.999966347530033,
            -0.001422739138722922,
            0.008079580483432283,
            0.001365741834644127,
            0.9999741760894847,
            0.007055629199258132,
            -0.008089410156878961,
            -0.007044357138835809,
            0.9999424675829176,
        ]
    ).reshape(3, 3)
    p_l = np.array(
        [
            435.2046959714599,
            0,
            367.4517211914062,
            0,
            0,
            435.2046959714599,
            252.2008514404297,
            0,
            0,
            0,
            1,
            0,
        ]
    ).reshape(3, 4)
    map_l = cv2.initUndistortRectifyMap(
        k_l, d_l, r_l, p_l[:3, :3], (752, 480), cv2.CV_32F
    )

    k_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(
        3, 3
    )
    d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0])
    r_r = np.array(
        [
            0.9999633526194376,
            -0.003625811871560086,
            0.007755443660172947,
            0.003680398547259526,
            0.9999684752771629,
            -0.007035845251224894,
            -0.007729688520722713,
            0.007064130529506649,
            0.999945173484644,
        ]
    ).reshape(3, 3)
    p_r = np.array(
        [
            435.2046959714599,
            0,
            367.4517211914062,
            -47.90639384423901,
            0,
            435.2046959714599,
            252.2008514404297,
            0,
            0,
            0,
            1,
            0,
        ]
    ).reshape(3, 4)
    map_r = cv2.initUndistortRectifyMap(
        k_r, d_r, r_r, p_r[:3, :3], (752, 480), cv2.CV_32F
    )

    intrinsics_vec = [435.2046959714599, 435.2046959714599, 367.4517211914062, 252.2008514404297]
    ht0, wd0 = 480, 752
    
    
    image_size = tuple(image_size)
    # Stride is used here
    images_left = sorted(glob.glob(os.path.join(datapath, "mav0/cam0/data/*.png")))[::stride]
    images_right = [x.replace("cam0", "cam1") for x in images_left]

    data = []
    for img_l, img_r in zip(images_left, images_right):
        if stereo and not os.path.isfile(img_r):
            continue

        timestamp = float(Path(img_l).stem)
        images = [cv2.remap(cv2.imread(img_l), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]

        if stereo:
            images.append(
                cv2.remap(cv2.imread(img_r), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)
            )

        images = [cv2.resize(image, (image_size[1], image_size[0])) for image in images]
        images = torch.from_numpy(np.stack(images, 0)).permute(0, 3, 1, 2).to(torch.float32)

        intrinsics = torch.as_tensor(intrinsics_vec)
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        data.append((timestamp, images, intrinsics))

    return data

# Causal Droid mirrors droid.py without the self.backend and self.traj_filler components. 
class CausalDroid:
    """DROID frontend-only runner: motion filter + online/local BA only."""

    def __init__(self, args):
        args.disable_vis = True  # force disable visualizer
        self.net = self.load_weights(args.weights)
        self.args = args
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)
        self.frontend = DroidFrontend(self.net, self.video, self.args)

        if not args.disable_vis:
            from visualizer.droid_visualizer import visualization_fn

            self.visualizer = torch.multiprocessing.Process(
                target=visualization_fn, args=(self.video, None)
            )
            self.visualizer.start()

    # Mirrors the load_weights method in droid.py but has cosmetic changes
    @staticmethod
    def load_weights(weights):
        net = DroidNet()
        state_dict = OrderedDict(
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()
        )

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        net.load_state_dict(state_dict)
        return net.to("cuda:0").eval()

    # Same as droid.py 
    def track(self, timestamp, image, depth=None, intrinsics=None):
        with torch.no_grad():
            # Decide decides whether the frame becomes a keyframe if 
            # predicted flow magnitude is larger than filter_thresh
            self.filterx.track(timestamp, image, depth, intrinsics)
            self.frontend()

    # Replaces Droid.terminate
    def keyframe_trajectory(self):
        with self.video.get_lock():
            n = self.video.counter.value
            timestamps = self.video.tstamp[:n].detach().cpu().numpy().astype(np.float64)
            poses = self.video.poses[:n].clone()

        traj = lietorch.SE3(poses).inv().data.detach().cpu().numpy()
        return timestamps, traj

    # Dense trajectory via PoseTrajectoryFiller: SE(3) screw interpolation
    # between bracketing keyframes + 6 iters of motion-only BA per non-keyframe.
    # Keyframe poses in self.video are NOT modified (no global BA).
    def filled_trajectory(self, image_stream_list):
        timestamps = np.asarray(
            [t for (t, _, _) in image_stream_list], dtype=np.float64
        )
        filler = PoseTrajectoryFiller(self.net, self.video)
        camera_trajectory = filler(image_stream_list)
        traj = camera_trajectory.inv().data.detach().cpu().numpy()
        return timestamps, traj

    def close(self):
        if hasattr(self, "visualizer"):
            self.visualizer.terminate()
            self.visualizer.join()


def save_euroc_trajectory(path, timestamps, traj):
    """Save timestamp tx ty tz qw qx qy qz, matching EuRoC ASL pose files."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("#timestamp [ns] p_RS_R_x [m] p_RS_R_y [m] p_RS_R_z [m] ")
        f.write("q_RS_w [] q_RS_x [] q_RS_y [] q_RS_z []\n")

        for timestamp, pose in zip(timestamps, traj):
            stamp = f"{timestamp:.0f}" if abs(timestamp) > 1e12 else f"{timestamp:.9f}"
            tx, ty, tz, qx, qy, qz, qw = pose.tolist()
            values = [stamp, tx, ty, tz, qw, qx, qy, qz]
            f.write(" ".join(str(v) if isinstance(v, str) else f"{v:.9f}" for v in values))
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Causal EuRoC trajectory export for DROID-SLAM: motion filter + "
            "frontend/local BA only (no backend, no trajectory filler)."
        )
    )
    parser.add_argument("--datapath", required=True, help="path to one EuRoC sequence")
    parser.add_argument("--save_path", required=True, help="EuRoC-format output trajectory path (keyframes only)")
    parser.add_argument(
        "--save_path_filled",
        help=(
            "optional EuRoC-format output path for the dense trajectory produced "
            "by the trajectory filler (per-frame motion-only BA, no global BA)"
        ),
    )
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", type=int, nargs=2, default=[320, 512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--stride", type=int, default=1, help="raw input stride applied when building the image list (filler operates on this list)")
    parser.add_argument("--track_stride", type=int, default=1, help="within the strided list, feed every Nth frame to the tracker; filler still uses every frame")

    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--disable_vis", dest="disable_vis", action="store_true", default=True)
    vis.add_argument("--enable_vis", dest="disable_vis", action="store_false")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.0)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--upsample", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.set_start_method("spawn")

    scene = Path(args.datapath).name
    print(f"Running causal frontend-only EuRoC evaluation on {args.datapath}")
    print(args)

    droid = CausalDroid(args)
    images = image_stream(args.datapath, image_size=args.image_size, stereo=args.stereo, stride=args.stride)
    track_images = images[::args.track_stride]

    filled = None
    try:
        for timestamp, image, intrinsics in tqdm(track_images, desc=scene):
            droid.track(timestamp, image, intrinsics=intrinsics)

        est_ts, est_traj = droid.keyframe_trajectory()
        save_euroc_trajectory(args.save_path, est_ts, est_traj)
        print(f"Saved causal EuRoC keyframe trajectory to {args.save_path}")

        if args.save_path_filled:
            filled = droid.filled_trajectory(images)
            filled_ts, filled_traj = filled
            save_euroc_trajectory(args.save_path_filled, filled_ts, filled_traj)
            print(f"Saved filled EuRoC trajectory to {args.save_path_filled}")
    finally:
        droid.close()
