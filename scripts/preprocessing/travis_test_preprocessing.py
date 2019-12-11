from avod.builders.dataset_builder import DatasetBuilder

from scripts.preprocessing import gen_mini_batches
from scripts.preprocessing import gen_label_clusters


def main():

    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_UNITTEST)
    dataset_config.data_split = "trainval"

    dataset_config.kitti_utils_config.pred_anchors_dir = '/home/amax_yly/Dataset/kitti/training/estimate3d_use_gt_simple' 
    dataset_config.kitti_utils_config.use_grid_anchors = False 
    unittest_dataset = DatasetBuilder.build_kitti_dataset(dataset_config)

    gen_label_clusters.main(unittest_dataset)
    gen_mini_batches.main(unittest_dataset)


if __name__ == '__main__':
    main()
