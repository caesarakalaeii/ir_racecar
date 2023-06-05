import cv2

#~ indicates a private parameter and will adjust to the namespace

default_list = {

        "camera1": {
            "ros_param": "~camera1",
            "default": "/joined_cams/usb_cam1/image_mono"
            },
        "camera2": {
            "ros_param": "~camera2",
            "default": "/joined_cams/usb_cam2/image_mono"
            },
        "publish": {
            "ros_param": "~publish",
            "default": "joined_image"
            },
        "queue_size": {
            "ros_param": "~queue_size",
            "default": 10
            },
        "encoding": {
            "ros_param": "~encoding",
            "default": "bgr8"
            },
        "verbose": {
            "ros_param": "~verbose",
            "default": False
            },
        "join_type": {
            "ros_param": "~join_type",
            "default": 1
            },
        "left_y_offset": {
            "ros_param": "~left_y_offset",
            "default": 20
            },
        "right_y_offset": {
            "ros_param": "~right_y_offset",
            "default": 0
            },
        "left_x_offset": {
            "ros_param": "~left_x_offset",
            "default": 0
            },
        "right_x_offset": {
            "ros_param": "~right_x_offset",
            "default": 0
            },
        "ratio": {
            "ros_param": "~ratio",
            "default": 0.85
            },
        "min_match": {
            "ros_param": "~min_match",
            "default": 10
            },
        "smoothing_window_size": {
            "ros_param": "~smoothing_window_size",
            "default": 50
            },
        "matching_write": {
            "ros_param": "~matching_write",
            "default": False
            },
        "static_matrix": {
            "ros_param": "~static_matrix",
            "default": False
            },
        "static_mask": {
            "ros_param": "~static_mask",
            "default": False
            },
        "stitchter_type": {
            "ros_param": "~stitchter_type",
            "default": cv2.Stitcher_PANORAMA
            },
        "direct_import": {
            "ros_param": "~direct_import",
            "default": False
            },
        "direct_import_sources": {
            "ros_param": "~direct_import_sources",
            "default": (0,
            )
            },
        "timing": {
            "ros_param": "~timing",
            "default": False
            },
        "ros_log": {
            "ros_param": "~ros_log",
            "default": False
            },
        "console_log": {
            "ros_param": "~console_log",
            "default": False
            },
        "cuda_device": {
            "ros_param": "~cuda_device",
            "default": 0
            },
        "try_cuda": {
            "ros_param": "~try_cuda",
            "default": True
            },
        "work_megapix": {
            "ros_param": "~work_megapix",
            "default": 0.6
            },
        "features": {
            "ros_param": "~features",
            "default": "orb" #best for CUDA: orb
            },
        "matcher": {
            "ros_param": "~matcher",
            "default": "homography"
            },
        "estimator": {
            "ros_param": "~estimator",
            "default": "homography" #other choice is affine
            },
        "match_conf": {
            "ros_param": "~match_conf",
            "default": 0.65
            },
        "conf_thresh": {
            "ros_param": "~conf_thresh",
            "default": 0.3 # was 1.0, lower to make blending() more reliable
            },
        "ba": {
            "ros_param": "~ba",
            "default": 'ray' # other : 'reproj','affine','no'
            },
        "ba_refine_mask": {
            "ros_param": "~ba_refine_mask",
            "default": "xxxxx"
            },
        "wave_correct": {
            "ros_param": "~wave_correct",
            "default": "horiz" #other : "vert", "no"
            },
        "save_graph": {
            "ros_param": "~save_graph",
            "default": None
            },
        "warp": {
            "ros_param": "~warp",
            "default": "plane" #WARP_CHOICES = ('spherical','plane','affine','cylindrical','fisheye','stereographic','compressedPlaneA2B1','compressedPlaneA1.5B1','compressedPlanePortraitA2B1','compressedPlanePortraitA1.5B1','paniniA2B1','paniniA1.5B1','paniniPortraitA2B1','paniniPortraitA1.5B1','mercator','transverseMercator'
            },
        "seam_megapix": {
            "ros_param": "~seam_megapix",
            "default": 0.1
            },
        "seam": {
            "ros_param": "~seam",
            "default": 'gc_color' #'gc_color','gc_colorgrad','dp_color','dp_colorgrad','voronoi','no'
            },
        "compose_megapix": {
            "ros_param": "~compose_megapix",
            "default": -1
            },
        "expos_comp": {
            "ros_param": "~expos_comp",
            "default": 'gain_blocks' # other :'gain','channel','channel_blocks','no'
            },
        "expos_comp_nr_feeds": {
            "ros_param": "~expos_comp_nr_feeds",
            "default": 1
            },
        "expos_comp_nr_filtering": {
            "ros_param": "~expos_comp_nr_filtering",
            "default": 2
            },
        "expos_comp_block_size": {
            "ros_param": "~expos_comp_block_size",
            "default": 32
            },
        "blend": {
            "ros_param": "~blend",
            "default": 'multiband' # other: 'feather', 'no'
            },
        "blend_strength": {
            "ros_param": "~blend_strength",
            "default": 5
            },
        "output": {
            "ros_param": "~output",
            "default": "result.jpg" #left over from stiching detailed, may be removed later
            },
        "timelapse": {
            "ros_param": "~timelapse",
            "default": None
            },
        "rangewidth": {
            "ros_param": "~rangewidth",
            "default": -1}
    }
