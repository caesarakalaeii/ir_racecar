'''
Utility class to return ImageJoin opjects depending on the set Parameters in arg_dict.
Reference parameters.py for usage of arg_dict
'''

from parameters import default_list
from image_join_hconcat import ImageJoinHConcat
from image_join_feature import ImageJoinFeature
from image_join_opencv import ImageJoinOpenCV
from image_join_cuda import ImageJoinCuda

class ImageJoinFactory():
    

    def create_instance(arg_dict):
        for k, v in default_list.items():
            if k in arg_dict:
                continue
            else: arg_dict.update({k:v["default"]})
        joinType = arg_dict["join_type"]
        if joinType == 1:
            return ImageJoinHConcat(arg_dict["left_y_offset"],
                                    arg_dict["right_y_offset"],
                                    arg_dict["left_x_offset"],
                                    arg_dict["right_x_offset"],
                                    arg_dict["logger"])
        elif joinType == 2:
            return ImageJoinFeature(ratio=arg_dict["ratio"],
                                    min_match=arg_dict["min_match"],
                                    smoothing_window_size=arg_dict["smoothing_window_size"],
                                    matching_write=arg_dict["matching_write"],
                                    static_matrix=arg_dict["static_matrix"],
                                    static_mask=arg_dict["static_mask"],
                                    logger=arg_dict["logger"],
                                    finder=arg_dict["finder"],
                                    matcher=arg_dict["matcher"])
        elif joinType == 3:
            return ImageJoinOpenCV(arg_dict["ratio"],
                                   arg_dict["min_match"],
                                   arg_dict["smoothing_window_size"],
                                   arg_dict["stitchter_type"],
                                   arg_dict["logger"])
        elif joinType == 4:
            return ImageJoinCuda()
        
        else:
            raise ValueError("JoinType not known, please use either CONCAT = 1, FEATURE = 2, OPENCV = 3, CUDA = 4")