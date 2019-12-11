"""Tests for avod.core.models.rpn_bev"""

import numpy as np
import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_build
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.rpn_model_bev import RpnModelBev
from avod.protos import pipeline_pb2


class RpnModelTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        pipeline_config = pipeline_pb2.NetworkPipelineConfig()
        dataset_config = pipeline_config.dataset_config
        config_path = avod.root_dir() + '/configs/unittest_model.config'

        cls.model_config = config_build.get_model_config_from_file(config_path)

        dataset_config.MergeFrom(DatasetBuilder.KITTI_UNITTEST)
        dataset_config.kitti_utils_config.pred_anchors_dir = '/home/amax_yly/Dataset/kitti/training/estimate3d_use_gt_simple' 
        dataset_config.kitti_utils_config.use_grid_anchors = False 
        cls.dataset = DatasetBuilder.build_kitti_dataset(dataset_config)

    def test_rpn_loss(self):
        # Use "val" so that the first sample is loaded each time
        rpn_model = RpnModelBev(self.model_config,
                             train_val_test="val",
                             dataset=self.dataset)

        predictions = rpn_model.build()

        loss, total_loss = rpn_model.loss(predictions)

        feed_dict = rpn_model.create_feed_dict()

        with self.test_session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            loss_dict_out = sess.run(loss, feed_dict=feed_dict)
            print('Losses ', loss_dict_out)

if __name__ == '__main__':
    tf.test.main()
