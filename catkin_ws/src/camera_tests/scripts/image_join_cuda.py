'''
Modified Version of OpenCVs sample stiching_detailed.py
Not yet fully working, highly unstable

'''


from image_join import ImageJoin
import cv2 as cv
import numpy as np
from collections import OrderedDict
from parameters import default_list

class ImageJoinCuda(ImageJoin):
    def create_EXPOS_COMP_CHOICES(self):
        self.EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
        self.EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
        self.EXPOS_COMP_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
        self.EXPOS_COMP_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
        self.EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO
    def create_BA_COST_CHOICES(self):
        self.BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
        self.BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
        self.BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
        self.BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster
    def create_FEATURES_FIND_CHOICES(self):
        try:
            cv.xfeatures2d_SURF.create() # check if the function can be called_FIND_CHOICES['surf'] = cv.xfeatures2d_SURF.create
        except (AttributeError, cv.error) as e:
            print("SURF not available")
        # if SURF not available, ORB is defaul_FIND_CHOICES['orb'] = cv.ORB.create
        try:self.FEATURES_FIND_CHOICES['sift'] = cv.SIFT_create
        except AttributeError:
            print("SIFT not available")
        try:self.FEATURES_FIND_CHOICES['brisk'] = cv.BRISK_create
        except AttributeError:
            print("BRISK not available")
        try:self.FEATURES_FIND_CHOICES['akaze'] = cv.AKAZE_create
        except AttributeError:
            print("AKAZE not available")
            
    def create_SEAM_FIND_CHOICES(self):
        self.SEAM_FIND_CHOICES['gc_color'] = cv.detail_GraphCutSeamFinder('COST_COLOR')
        self.SEAM_FIND_CHOICES['gc_colorgrad'] = cv.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
        self.SEAM_FIND_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
        self.SEAM_FIND_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
        self.SEAM_FIND_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
        self.SEAM_FIND_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)

    def create_ESTIMATOR_CHOICES(self):
        self.ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator
        self.ESTIMATOR_CHOICES['affine'] = cv.detail_AffineBasedEstimator
        
    def create_WAVE_CORRECT_CHOICES(self):
        self.WAVE_CORRECT_CHOICES['horiz'] = cv.detail.WAVE_CORRECT_HORIZ
        self.WAVE_CORRECT_CHOICES['no'] = None
        self.WAVE_CORRECT_CHOICES['vert'] = cv.detail.WAVE_CORRECT_VERT
            
    def __init__(self, arg_dict):
        super().__init__(arg_dict["logger"])
        self.EXPOS_COMP_CHOICES = OrderedDict()
        self.create_EXPOS_COMP_CHOICES()

        self.BA_COST_CHOICES = OrderedDict()
        self.create_BA_COST_CHOICES()
        self.FEATURES_FIND_CHOICES = OrderedDict()
        self.create_FEATURES_FIND_CHOICES()

        self.SEAM_FIND_CHOICES = OrderedDict()
        self.create_SEAM_FIND_CHOICES()
        
        self.ESTIMATOR_CHOICES = OrderedDict()
        self.create_ESTIMATOR_CHOICES()

        self.WARP_CHOICES = (
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

        self.WAVE_CORRECT_CHOICES = OrderedDict()
        self.create_WAVE_CORRECT_CHOICES()

        self.BLEND_CHOICES = ('multiband', 'feather', 'no',)
        
        for k,v in default_list.items():
            if not k in arg_dict:
                arg_dict.update({k:v["default"]})
        self.arg_dict = arg_dict
        
        
        self.work_megapix = self.arg_dict["work_megapix"]
        self.seam_megapix = self.arg_dict["seam_megapix"]
        self.compose_megapix = self.arg_dict["compose_megapix"]
        self.conf_thresh = self.arg_dict["conf_thresh"]
        self.ba_refine_mask = self.arg_dict["ba_refine_mask"]
        self.wave_correct = self.WAVE_CORRECT_CHOICES[self.arg_dict["wave_correct"]]
        if self.arg_dict["save_graph"] is None:
            self.save_graph = False
        else:
            self.save_graph = True
        self.warp_type = self.arg_dict["warp"]
        self.blend_type = self.arg_dict["blend"]
        self.blend_strength = self.arg_dict["blend_strength"]
        self.result_name = self.arg_dict["output"]
        if self.arg_dict["timelapse"]:
            self.timelapse = True
            if self.arg_dict["timelapse"] == "as_is":
                self.timelapse_type = cv.detail.Timelapser_AS_IS
            elif self.arg_dict["timelapse"] == "crop":
                self.timelapse_type = cv.detail.Timelapser_CROP
            else:
                self.l.fail("Bad timelapse method")
                exit()
        else:
            self.timelapse = False
        self.matcher = None
        self.finder  = self.FEATURES_FIND_CHOICES[self.arg_dict["features"]]()
        self.seam_work_aspect = 1
        self.compensator = None
        
        
        
        

    def blending(self, img1, img2):
        frames = [img1, img2]
        i=0
        img_names = [] #variable left over from OpenCVs sample stitching_detailed may be removed later
        full_img_sizes = []
        images = []
        features = []
        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False
        for frame in frames:
            full_img = frame
            img_names.append("Frame %d"%i) #simulate existence of file names
            i+=1
            if full_img is None:
                print("Cannot read Frame ", name)
                exit()
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            if self.work_megapix < 0:
                img = full_img
                work_scale = 1
                is_work_scale_set = True
            else:
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(self.work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    is_work_scale_set = True
                img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
            if is_seam_scale_set is False:
                if self.seam_megapix > 0:
                    seam_scale = min(1.0, np.sqrt(self.seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                else:
                    seam_scale = 1.0
                seam_work_aspect = seam_scale / work_scale
                is_seam_scale_set = True
            img_feat = cv.detail.computeImageFeatures2(self.finder, img)
            features.append(img_feat)
            img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
            images.append(img)
        if self.matcher is None:
            self.set_matcher()
        p = self.matcher.apply2(features)
        self.matcher.collectGarbage()

        if self.save_graph:
            with open(self.arg_dict["save_graph"], 'w') as fh:
                fh.write(cv.detail.matchesGraphAsString(img_names, p, self.conf_thresh))

        indices = cv.detail.leaveBiggestComponent(features, p, self.conf_thresh)
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
            raise AttributeError ("Need more images")

        estimator = self.ESTIMATOR_CHOICES[self.arg_dict["estimator"]]()
        b, cameras = estimator.apply(features, p, None)
        if not b:
            raise AssertionError("Homography estimation failed.")
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = self.BA_COST_CHOICES[self.arg_dict["ba"]]()
        adjuster.setConfThresh(self.conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        if self.ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if self.ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if self.ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if self.ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if self.ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)
        b, cameras = adjuster.apply(features, p, cameras)
        if not b:
            raise AssertionError("Camera parameters adjusting failed.")
        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        focals.sort()
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        if self.wave_correct is not None:
            rmats = []
            for cam in cameras:
                rmats.append(np.copy(cam.R))
            rmats = cv.detail.waveCorrect(rmats, self.wave_correct)
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

        warper = cv.PyRotationWarper(self.warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
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
        if self.compensator is None:
            self.set_compensator()
        self.compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        seam_finder = self.SEAM_FIND_CHOICES[self.arg_dict["seam"]]
        masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
        compose_scale = 1
        corners = []
        sizes = []
        blender = None
        timelapser = None
        # https://github.com/opencv/opencv/blob/4.x/samples/cpp/stitching_detailed.cpp#L725 ?
        for idx, name in enumerate(img_names):
            full_img = frame
            if not is_compose_scale_set:
                if self.compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(self.compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_scale
                warped_image_scale *= compose_work_aspect
                warper = cv.PyRotationWarper(self.warp_type, warped_image_scale)
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
            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT) #Every Iteration
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT) #every iteration
            self.compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv.dilate(masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            if blender is None and not self.timelapse:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength / 100
                if blend_width < 1:
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif self.blend_type == "multiband":
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
                elif self.blend_type == "feather":
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)
            elif timelapser is None and self.timelapse:
                timelapser = cv.detail.Timelapser_createDefault(self.timelapse_type)
                timelapser.initialize(corners, sizes)
            if self.timelapse:
                ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
                timelapser.process(image_warped_s, ma_tones, corners[idx])
                pos_s = img_names[idx].rfind("/")
                if pos_s == -1:
                    fixed_file_name = "fixed_" + img_names[idx]
                else:
                    fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
                cv.imwrite(fixed_file_name, timelapser.getDst()) #TODO still writing, turn into return if timelapser is needed
            else:
                blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx]) #every iteration
        if not self.timelapse:
            result = None
            result_mask = None
            result, result_mask = blender.blend(result, result_mask)
            zoom_x = 600.0 / result.shape[1]
            dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
            return dst
    
    def set_matcher(self):
        try_cuda = self.arg_dict["try_cuda"]
        matcher_type = self.arg_dict["matcher"]
        if self.arg_dict["match_conf"] is None:
            if self.arg_dict["features"] == 'orb':
                match_conf = 0.3
            else:
                match_conf = 0.65
        else:
            match_conf = self.arg_dict["match_conf"]
        range_width = self.arg_dict["rangewidth"]
        if matcher_type == "affine":
            matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
        elif range_width == -1:
            matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)
        else:
            matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda, match_conf)
        self.matcher = matcher


    def set_compensator(self):
        expos_comp_type = self.EXPOS_COMP_CHOICES[self.arg_dict["expos_comp"]]
        expos_comp_nr_feeds = self.arg_dict["expos_comp_nr_feeds"]
        expos_comp_block_size = self.arg_dict["expos_comp_block_size"]
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
        self.compensator = compensator
