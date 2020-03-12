"""
Generates 3D anchors, placing them on the ground plane
"""

import numpy as np

from avod.core import anchor_generator
from avod.core import box_3d_encoder
from avod.core import box_8c_encoder
from wavedata.tools.obj_detection import obj_utils


class PredAnchor3dGenerator(anchor_generator.AnchorGenerator):

    def name_scope(self):
        return 'PredAnchor3dGenerator'

    def _generate(self, **params):
        """
        Generates 3D anchors from predicted result of 2D detection in the provided 3d area and places them on the ground_plane.

        Args:
            **params:
                area_3d: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]

        Returns:
            list of 3D anchors in the form N x [x, y, z, l, w, h, ry]
        """
        #pred_label_list = params.get('pred_label_list')
        pred_anchors_dir = params.get('pred_anchors_dir')
        sample_name = params.get('sample_name')
        class_name = params.get('class_name')
        return augment_anchors(sample_name, class_name, pred_anchors_dir)

    def generate_aug(self, **params):
        """
        Generates 3D anchors from predicted result of 2D detection in the provided 3d area and places them on the ground_plane.

        Args:
            **params:
                area_3d: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]

        Returns:
            list of 3D anchors in the form N x [x, y, z, l, w, h, ry]
        """
        #pred_label_list = params.get('pred_label_list')
        pred_anchors_dir = params.get('pred_anchors_dir')
        sample_name = params.get('sample_name')
        class_name = params.get('class_name')
        return augment_anchors(sample_name, class_name, pred_anchors_dir, aug_shape=True)




def augment_anchors(sample_name, class_name, pred_anchors_dir, aug_shape=False):

    #pred_anchors_dir = '/home/amax_yly/Dataset/kitti/training/estimate3d_use_gt_simple'
    #estimated_label_dir = '/home/amax_yly/Dataset/kitti/training/estimate3d_use_gt_simple'
    img_idx = int(sample_name)
    pred_label_list = obj_utils.read_labels(pred_anchors_dir, img_idx) 

    filtered_pred_list = pred_label_list
    #filtered_pred_list = [obj for obj in pred_label_list if obj.type in [class_name]]
    #filtered_pred_list = dataset_utils.filter_labels(pred_label_list, classes=[class_name]])
    all_anchors = [] 
    for label in filtered_pred_list:
        box = box_3d_encoder.object_label_to_box_3d(label)
        # add augmentation
        augmented = augment_box(box, aug_shape)
        all_anchors.append(augmented)

    all_anchors = np.asarray(all_anchors)
    if len(all_anchors) > 0:
        all_anchors = np.reshape(all_anchors, (-1, all_anchors.shape[-1]))
    return all_anchors 

def augment_box(box, aug_shape):
    """Augment 3D box
    camera coordinate, x, y, z
    box: (x, y, z, l, w, h, ry)
    Returns (num_aug, 7)
    """
    x = box[0]
    z = box[2]
    
    step = 1
    offset = np.arange(-15, 15, step)

    offset_x = np.abs(x)/np.sqrt(x**2+z**2) * offset
    offset_z = offset_x * (0 - z) / (0 - x)

    aug_translated = []
    #append original predicted 3D box to be first one 
    aug_translated.append(box)   
    for i, dx in enumerate(offset_x):
        if i == len(offset)/2:
            continue   #origin one has already been appended.
        newbox = box.copy()
        newbox[0] += dx
        newbox[2] += offset_z[i] 

        if aug_shape:
            # add at length and width
            newbox_8c = box_8c_encoder.np_box_3d_to_box_8co(newbox)
            xmin, zmin = np.min(newbox_8c[[0, 2], :], 1)
            xmax, zmax = np.max(newbox_8c[[0, 2], :], 1)
            newbox[3] = xmax - xmin
            newbox[4] = zmax - zmin

        aug_translated.append(newbox)

    aug_translated = np.array(aug_translated)
    
    #offset_theta = args.aug_rotate_param.split(',')
    #offset_theta = np.array([float(t) for t in offset_theta])
    #print('[aug box]offset theta ',offset_theta)
    offset_theta = np.array([-3., -1.5, 1.5, 3.])
    #angle to rad
    offset_theta = offset_theta / 180 * np.pi
    aug_rotated = []
    #add rotate for each translated box
    for dtheta in offset_theta:
        cos = np.cos(dtheta)
        sin = np.sin(dtheta)
        rot = np.array([[cos, -sin],[sin, cos]])
        rot=np.squeeze(rot)
        rp = rot.dot(np.array([x, z]))
        xt = rp[0]
        zt = rp[1]
        offset_x = np.abs(xt)/np.sqrt(xt**2+zt**2) * offset
        offset_z = offset_x * (0 - zt)/(0-xt)
        newboxes_dt = aug_translated.copy()
        newboxes_dt[:, 0] = xt + offset_x
        newboxes_dt[:, 2] = zt + offset_z
        aug_rotated.append(newboxes_dt)

    aug_rotated = np.array(aug_rotated)
    aug_rotated = aug_rotated.reshape((-1, 7))

    #newboxes = aug_translated
    newboxes = np.vstack([aug_translated, aug_rotated])
   
    #print(newboxes.shape)
    return newboxes 


def tile_anchors_3d(area_extents,
                    anchor_3d_sizes,
                    anchor_stride,
                    ground_plane):
    """
    Tiles anchors over the area extents by using meshgrids to
    generate combinations of (x, y, z), (l, w, h) and ry.

    Args:
        area_extents: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        anchor_3d_sizes: list of 3d anchor sizes N x (l, w, h)
        anchor_stride: stride lengths (x_stride, z_stride)
        ground_plane: coefficients of the ground plane e.g. [0, -1, 0, 0]

    Returns:
        boxes: list of 3D anchors in box_3d format N x [x, y, z, l, w, h, ry]
    """
    # Convert sizes to ndarray
    anchor_3d_sizes = np.asarray(anchor_3d_sizes)

    anchor_stride_x = anchor_stride[0]
    anchor_stride_z = anchor_stride[1]
    anchor_rotations = np.asarray([0, np.pi / 2.0])

    x_start = area_extents[0][0] + anchor_stride[0] / 2.0
    x_end = area_extents[0][1]
    x_centers = np.array(np.arange(x_start, x_end, step=anchor_stride_x),
                         dtype=np.float32)

    z_start = area_extents[2][1] - anchor_stride[1] / 2.0
    z_end = area_extents[2][0]
    z_centers = np.array(np.arange(z_start, z_end, step=-anchor_stride_z),
                         dtype=np.float32)

    # Use ranges for substitution
    size_indices = np.arange(0, len(anchor_3d_sizes))
    rotation_indices = np.arange(0, len(anchor_rotations))

    # Generate matrix for substitution
    # e.g. for two sizes and two rotations
    # [[x0, z0, 0, 0], [x0, z0, 0, 1], [x0, z0, 1, 0], [x0, z0, 1, 1],
    #  [x1, z0, 0, 0], [x1, z0, 0, 1], [x1, z0, 1, 0], [x1, z0, 1, 1], ...]
    before_sub = np.stack(np.meshgrid(x_centers,
                                      z_centers,
                                      size_indices,
                                      rotation_indices),
                          axis=4).reshape(-1, 4)

    # Place anchors on the ground plane
    a, b, c, d = ground_plane
    all_x = before_sub[:, 0]
    all_z = before_sub[:, 1]
    all_y = -(a * all_x + c * all_z + d) / b

    # Create empty matrix to return
    num_anchors = len(before_sub)
    all_anchor_boxes_3d = np.zeros((num_anchors, 7))

    # Fill in x, y, z
    all_anchor_boxes_3d[:, 0:3] = np.stack((all_x, all_y, all_z), axis=1)

    # Fill in shapes
    sizes = anchor_3d_sizes[np.asarray(before_sub[:, 2], np.int32)]
    all_anchor_boxes_3d[:, 3:6] = sizes

    # Fill in rotations
    rotations = anchor_rotations[np.asarray(before_sub[:, 3], np.int32)]
    all_anchor_boxes_3d[:, 6] = rotations

    return all_anchor_boxes_3d
