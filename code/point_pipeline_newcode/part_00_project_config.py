import os

DATASET_ALIASES = {
    "loop": "tum_loop",
    "rgbd": "tum_rgbd",
    "xyz": "rgbd_dataset_freiburg1_xyz",
    "360": "rgbd_dataset_freiburg1_360",
    "room": "rgbd_dataset_freiburg1_room",
    "fr2desk": "rgbd_dataset_freiburg2_desk",
    "desk2": "rgbd_dataset_freiburg1_desk2",
    "floor": "rgbd_dataset_freiburg1_floor",
}

dataset_key = os.environ.get("SLAM_DATASET", "").strip().lower()
DATASET_DIR = DATASET_ALIASES.get(dataset_key, os.environ.get("SLAM_DATASET_DIR", "tum_loop"))
RGB_TXT = os.path.join(DATASET_DIR, "rgb.txt")
DEPTH_TXT = os.path.join(DATASET_DIR, "depth.txt")
GT_PATH = os.path.join(DATASET_DIR, "groundtruth.txt")

OUTPUT_DIR = os.environ.get("SLAM_OUTPUT_DIR", "outputs")
if os.path.basename(os.path.normpath(OUTPUT_DIR)).lower() != "point_only":
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "point_only")
PREFIX = "icp_ransac"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if "freiburg1" in DATASET_DIR.lower():
    DATASET_PROFILE = "fr1"
    fx = 517.3
    fy = 516.5
    cx = 318.6
    cy = 255.3
else:
    DATASET_PROFILE = "fr2"
    fx = 520.9
    fy = 521.0
    cx = 325.1
    cy = 249.7

depth_scale = 5000.0

FRAME_STEP = 5
MAX_FRAMES = None

NFEATURES = 5000
RATIO_THRESH = 0.75
RANSAC_THRESH = 3.0
MIN_3D_MATCHES = 8
MIN_ODOM_RANSAC_INLIERS = 20

LK_MAX_CORNERS = 1200
LK_QUALITY_LEVEL = 0.01
LK_MIN_DISTANCE = 7
LK_BLOCK_SIZE = 7
LK_WIN_SIZE = 21
LK_MAX_LEVEL = 3
LK_FB_THRESH = 1.5

PIXEL_STEP = 4
MAX_DEPTH_M = 4.0
VOXEL_SIZE = 0.03

ICP_THRESH = 0.08
MIN_ICP_FITNESS = 0.30
MAX_ICP_RMSE = 0.05
MAX_TRANSLATION_JUMP = 0.20

LOOP_HEAD_FRAMES = 10
LOOP_QUERY_START_FRACTION = 0.5
LOOP_QUERY_STRIDE = 1
MIN_LOOP_SEPARATION = 8
MIN_LOOP_RANSAC_INLIERS = 20
MIN_LOOP_SCORE = 15.0

MAX_ASSOC_DIFF = 0.02
MAX_GT_DIFF = 0.05
