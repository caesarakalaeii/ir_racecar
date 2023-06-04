from __future__ import print_function
import numpy as np
from collections import OrderedDict
import cv2 as cv
import parameters as p



class ImageJoin(object):
    

    def __init__(self):
        pass

    def blending(self, img1, img2):
        pass

class ImageJoinFactory():
    

    def create_instance(dict):
        for k, v in p.default_list.items():
            if k in dict:
                continue
            else: dict.update({k:v})
        joinType = dict["join_type"]
        if joinType == 1:
            return ImageJoinHConcat(dict["left_y_offset"],
                                    dict["right_y_offset"],
                                    dict["left_x_offset"],
                                    dict["right_x_offset"])
        elif joinType == 2:
            return ImageJoinFeature(dict["ratio"],
                                    dict["min_match"],
                                    dict["smoothing_window_size"],
                                    dict["matching_write"],
                                    dict["static_matrix"],
                                    dict["static_mask"])
        elif joinType == 3:
            return ImageJoinOpenCV(dict["ratio"],
                                   dict["min_match"],
                                   dict["smoothing_window_size"],
                                   dict["stitchter_type"])
        elif joinType == 4:
            return ImageJoinCuda(dict)
        
        else:
            raise ValueError("JoinType not known, please use either CONCAT = 1, FEATURE = 2, OPENCV = 3, CUDA = 4")

class ImageJoinHConcat(ImageJoin):

    def __init__(self, left_y_offset = 20, right_y_offset=0, left_x_offset=0, right_x_offset=0):
        super().__init__()
        self.left_y_offset = left_y_offset
        self.left_x_offset = left_x_offset
        self.right_y_offset = right_y_offset
        self.right_x_offset = right_x_offset


    def blending(self, img1, img2):
        
        shape1 = img1.shape
        shape2 = img2.shape
        if self.left_y_offset != 0: #add black bar on top / bottom of imgages
            y_offset = np.zeros((abs(self.left_y_offset),shape1[1], shape1[2]))
            if self.left_y_offset < 0:
                img1= np.append(img1, y_offset, axis=0) 
                img2= np.append(y_offset, img1, axis=0) #add buffer fo left/right image to preserve shape
            else:
                img1= np.append(y_offset, img1, axis=0)
                img2= np.append(img2, y_offset, axis=0)
        if self.right_y_offset != 0:
            y_offset = np.zeros((abs(self.right_y_offset),shape1[1], shape1[2]))
            if self.right_y_offset < 0:
                img2= np.append(img2, y_offset, axis=1)
                img1= np.append(y_offset, img1, axis=1)
            else:
                img2= np.append(y_offset, img1, axis=1)
                img1= np.append(y_offset, img1, axis=1)
        if self.left_x_offset != 0:         #cut part from left/right
            img1 = img1[:,:-self.left_x_offset,:]
        if self.right_x_offset != 0:
            img2 = img2[:,-self.right_x_offset:,:]
        img = np.append(img1, img2, axis=1) #stack horizontally
        return cv.convertScaleAbs(img)  #convert to bgr8 compatible type and return
    
class ImageJoinFeature(ImageJoin):

    def __init__(self, ratio=0.85, min_match=10, smoothing_window_size=50, matching_write = False, static_matrix = False, static_mask = False ) :
        self.ratio=ratio
        self.min_match=min_match
        try:
            self.sift=cv.SIFT_create() #maybe replace wir ORB or AKAZE
        except AttributeError:
            #for older versions of open cv
            try:
                self.sift=cv.xfeatures2d.SIFT_create()
            except AttributeError:
                print("Unsupported CV version, exiting")
                exit(1)
        
        self.smoothing_window_size=smoothing_window_size
        self.matching_write = matching_write
        self.static_matrix = static_matrix
        self.matrix_set = False
        self.static_mask = static_mask
        self.mask_set = False
        self.mask1 = None
        self.mask2 = None

        super().__init__()


    def registration(self,img1,img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        if self.matching_write:
            cv.imwrite('matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv.findHomography(image2_kp, image1_kp, cv.RANSAC,5.0)
            self.matrix_set = True
            if(self.static_matrix):
                self.H = H
                
            return H

    def create_mask(self,img1,img2,version, hasDepth = True):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        if not hasDepth:
            return cv.merge([mask])

        return cv.merge([mask, mask, mask])

    def blending(self,img1,img2):
        if(self.static_matrix and self.matrix_set):
            return self.blending_no_reg(img1, img2, self.H)
        self.H = self.registration(img1,img2)
        return self.blending_no_reg(img1, img2, self.H)
        
    
    def blending_no_reg(self,img1,img2, H):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        depth = 0
        try:
            depth = img1.shape[2]
        except:
            depth = 0
        if depth == 0:
            panorama1 = np.zeros((height_panorama, width_panorama))
            if self.static_mask and self.mask_set:
                mask1 = self.mask1
            else:
                mask1 = self.create_mask(img1,img2,version='left_image', hasDepth=False)
                self.mask1 =  mask1
            panorama1[0:img1.shape[0], 0:img1.shape[1]] = img1
            panorama1 *= mask1
            if self.static_mask and self.mask_set:
                mask2 = self.mask2
            else:
                mask2 = self.create_mask(img1,img2,version='right_image', hasDepth=False)
                self.mask2 = mask2
                self.mask_set = True
            try:
                panorama2 = cv.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
            except:
                raise Exception("Couldn't match images.")
            result=panorama1+panorama2

            rows, cols = np.where(result[:, :] != 0)
            min_row, max_row = np.min(rows), np.max(rows) + 1
            min_col, max_col = np.min(cols), np.max(cols) + 1
            final_result = result[min_row:max_row, min_col:max_col]

        else :
            panorama1 = np.zeros((height_panorama, width_panorama, depth))
            mask1 = self.create_mask(img1,img2,version='left_image')
            panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
            panorama1 *= mask1
            mask2 = self.create_mask(img1,img2,version='right_image')
            panorama2 = cv.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
            result=panorama1+panorama2

            rows, cols = np.where(result[:, :, 0] != 0)
            min_row, max_row = np.min(rows), np.max(rows) + 1
            min_col, max_col = np.min(cols), np.max(cols) + 1
            final_result = result[min_row:max_row, min_col:max_col, :]
        return cv.convertScaleAbs(final_result)


class ImageJoinOpenCV(ImageJoin):
    def __init__(self, ratio = 0.5, min_match=3, smoothing_window_size = 10, stitchter_type = cv.Stitcher_PANORAMA) :
        self.ratio=ratio
        self.min_match=min_match
        self.stich = cv.Stitcher.create(stitchter_type)
        self.smoothing_window_size=smoothing_window_size
        super().__init__()


    def blending(self,img1,img2):
        img = []
        img.append(img1)
        img.append(img2)
        result = self.stich.stitch(img)
        if result[0] == cv.Stitcher_OK:
            return result[1]
        else:
            raise Exception("Image stitching failed, is the overlap big enough?")
        

class ImageJoinCuda(ImageJoin):
    EXPOS_COMP_CHOICES = OrderedDict()
    EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
    EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
    EXPOS_COMP_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
    EXPOS_COMP_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
    EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO

    BA_COST_CHOICES = OrderedDict()
    BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
    BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
    BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
    BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster

    FEATURES_FIND_CHOICES = OrderedDict()
    try:
        cv.xfeatures2d_SURF.create() # check if the function can be called
        FEATURES_FIND_CHOICES['surf'] = cv.xfeatures2d_SURF.create
    except (AttributeError, cv.error) as e:
        print("SURF not available")
    # if SURF not available, ORB is default
    FEATURES_FIND_CHOICES['orb'] = cv.ORB.create
    try:
        FEATURES_FIND_CHOICES['sift'] = cv.SIFT_create
    except AttributeError:
        print("SIFT not available")
    try:
        FEATURES_FIND_CHOICES['brisk'] = cv.BRISK_create
    except AttributeError:
        print("BRISK not available")
    try:
        FEATURES_FIND_CHOICES['akaze'] = cv.AKAZE_create
    except AttributeError:
        print("AKAZE not available")

    SEAM_FIND_CHOICES = OrderedDict()
    SEAM_FIND_CHOICES['gc_color'] = cv.detail_GraphCutSeamFinder('COST_COLOR')
    SEAM_FIND_CHOICES['gc_colorgrad'] = cv.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
    SEAM_FIND_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
    SEAM_FIND_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
    SEAM_FIND_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
    SEAM_FIND_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)

    ESTIMATOR_CHOICES = OrderedDict()
    ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator
    ESTIMATOR_CHOICES['affine'] = cv.detail_AffineBasedEstimator

    WARP_CHOICES = (
        'spherical',
        'plane',
        'affine',
        'cylindrical',
        'fisheye',
        'stereographic',
        'compressedPlaneA2B1',
        'compressedPlaneA1.5B1',
        'compressedPlanePortraitA2B1',
        'compressedPlanePortraitA1.5B1',
        'paniniA2B1',
        'paniniA1.5B1',
        'paniniPortraitA2B1',
        'paniniPortraitA1.5B1',
        'mercator',
        'transverseMercator',
    )

    WAVE_CORRECT_CHOICES = OrderedDict()
    WAVE_CORRECT_CHOICES['horiz'] = cv.detail.WAVE_CORRECT_HORIZ
    WAVE_CORRECT_CHOICES['no'] = None
    WAVE_CORRECT_CHOICES['vert'] = cv.detail.WAVE_CORRECT_VERT

    BLEND_CHOICES = ('multiband', 'feather', 'no',)
    
    def __init__(self, args):
        
        for k,v in p.default_cuda_join.items():
            args.update({k:v})
        self.args = args
            
        super().__init__()

    def blending(self, img1, img2):
        frames = [img1, img2]
        img_names = []
        work_megapix = dict["work_megapix"]
        seam_megapix = dict["seam_megapix"]
        compose_megapix = dict["compose_megapix"]
        conf_thresh = dict["conf_thresh"]
        ba_refine_mask = dict["ba_refine_mask"]
        wave_correct = WAVE_CORRECT_CHOICES[dict["wave_correct"]]
        if dict["save_graph"] is None:
            save_graph = False
        else:
            save_graph = True
        warp_type = dict["warp"]
        blend_type = dict["blend"]
        blend_strength = dict["blend_strength"]
        result_name = dict["output"]
        if dict["timelapse"] is not None:
            timelapse = True
            if dict["timelapse"] == "as_is":
                timelapse_type = cv.detail.Timelapser_AS_IS
            elif dict["timelapse"] == "crop":
                timelapse_type = cv.detail.Timelapser_CROP
            else:
                print("Bad timelapse method")
                exit()
        else:
            timelapse = False
        finder = FEATURES_FIND_CHOICES[dict["features"]]()
        seam_work_aspect = 1
        full_img_sizes = []
        features = []
        images = []
        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False
        i=0
        for frame in frames:
            full_img = frame
            img_names.append("img%d"%i)
            i+=1
            if full_img is None:
                print("Cannot read image ", name)
                exit()
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            if work_megapix < 0:
                img = full_img
                work_scale = 1
                is_work_scale_set = True
            else:
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    is_work_scale_set = True
                img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
            if is_seam_scale_set is False:
                if seam_megapix > 0:
                    seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                else:
                    seam_scale = 1.0
                seam_work_aspect = seam_scale / work_scale
                is_seam_scale_set = True
            img_feat = cv.detail.computeImageFeatures2(finder, img)
            features.append(img_feat)
            img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
            images.append(img)

        matcher = get_matcher_dict(dict)
        p = matcher.apply2(features)
        matcher.collectGarbage()

        if save_graph:
            with open(dict["save_graph"], 'w') as fh:
                fh.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))

        indices = cv.detail.leaveBiggestComponent(features, p, conf_thresh)
        img_subset = []
        img_names_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            img_names_subset.append(img_names[indices[i]])
            img_subset.append(images[indices[i]])
            full_img_sizes_subset.append(full_img_sizes[indices[i]])
        images = img_subset
        img_names = img_names_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(frames)
        if num_images < 2:
            print("Need more images")
            exit()

        estimator = ESTIMATOR_CHOICES[dict["estimator"]]()
        b, cameras = estimator.apply(features, p, None)
        if not b:
            print("Homography estimation failed.")
            exit()
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = BA_COST_CHOICES[dict["ba"]]()
        adjuster.setConfThresh(conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        if ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)
        b, cameras = adjuster.apply(features, p, cameras)
        if not b:
            print("Camera parameters adjusting failed.")
            exit()
        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        focals.sort()
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        if wave_correct is not None:
            rmats = []
            for cam in cameras:
                rmats.append(np.copy(cam.R))
            rmats = cv.detail.waveCorrect(rmats, wave_correct)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]
        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []
        for i in range(0, num_images):
            um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
            masks.append(um)

        warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
        for idx in range(0, num_images):
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())

        images_warped_f = []
        for img in images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)

        compensator = get_compensator_dict(dict)
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        seam_finder = SEAM_FIND_CHOICES[dict["seam"]]
        masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
        compose_scale = 1
        corners = []
        sizes = []
        blender = None
        timelapser = None
        # https://github.com/opencv/opencv/blob/4.x/samples/cpp/stitching_detailed.cpp#L725 ?
        for idx, name in enumerate(img_names):
            full_img = cv.imread(name)
            if not is_compose_scale_set:
                if compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_scale
                warped_image_scale *= compose_work_aspect
                warper = cv.PyRotationWarper(warp_type, warped_image_scale)
                for i in range(0, len(img_names)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                        int(round(full_img_sizes[i][1] * compose_scale)))
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])
            if abs(compose_scale - 1) > 1e-1:
                img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)
            else:
                img = full_img
            _img_size = (img.shape[1], img.shape[0])
            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv.dilate(masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            if blender is None and not timelapse:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
                if blend_width < 1:
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif blend_type == "multiband":
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
                elif blend_type == "feather":
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)
            elif timelapser is None and timelapse:
                timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
                timelapser.initialize(corners, sizes)
            if timelapse:
                ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
                timelapser.process(image_warped_s, ma_tones, corners[idx])
                pos_s = img_names[idx].rfind("/")
                if pos_s == -1:
                    fixed_file_name = "fixed_" + img_names[idx]
                else:
                    fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
                cv.imwrite(fixed_file_name, timelapser.getDst())
            else:
                blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
        if not timelapse:
            result = None
            result_mask = None
            result, result_mask = blender.blend(result, result_mask)
            return result
    
    def get_matcher_dict(dict):
        try_cuda = dict["try_cuda"]
        matcher_type = dict["matcher"]
        if dict["match_conf"] is None:
            if dict["features"] == 'orb':
                match_conf = 0.3
            else:
                match_conf = 0.65
        else:
            match_conf = dict["match_conf"]
        range_width = dict["rangewidth"]
        if matcher_type == "affine":
            matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
        elif range_width == -1:
            matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)
        else:
            matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda, match_conf)
        return matcher


    def get_compensator_dict(dict):
        expos_comp_type = EXPOS_COMP_CHOICES[dict["expos_comp"]]
        expos_comp_nr_feeds = dict["expos_comp_nr_feeds"]
        expos_comp_block_size = dict["expos_comp_block_size"]
        # expos_comp_nr_filtering = args.expos_comp_nr_filtering
        if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
            compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
            # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
        elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
            compensator = cv.detail_BlocksChannelsCompensator(
                expos_comp_block_size, expos_comp_block_size,
                expos_comp_nr_feeds
            )
            # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
        else:
            compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
        return compensator
