"""Contains modified SENet model  to apply attention mechanism to bev and img feature_map.

Usage:
    outputs, end_points = BevVgg(inputs, layers_config)
"""

import tensorflow as tf


slim = tf.contrib.slim


class SamModule():
    def __init__(self):
        pass

    def build(self, bev_feature_maps_org, img_feature_maps_org, is_training):
        with tf.variable_scope('bev_img_cam'):
            # hb, wb, cb = tuple([dim.value for dim in bev_feature_maps_org.shape[1:4]])
            # hi, wi, ci = tuple([dim.value for dim in img_feature_maps_org.shape[1:4]])

            bev_feature_maps_avg = tf.reduce_sum(bev_feature_maps_org, axis=-1, keep_dims=True)     #7*7*1
            img_feature_maps_avg = tf.reduce_sum(img_feature_maps_org, axis=-1, keep_dims=True)     #7*7*1
            bev_img_feature_maps_avg = tf.concat([bev_feature_maps_avg, img_feature_maps_avg], axis=-1)     #7*7*2
            net = slim.repeat(bev_img_feature_maps_avg,             # -> 5*5*2 -> 3*3*2
                              2,
                              slim.conv2d,
                              2,
                              [3, 3],
                              padding='VALID',
                              normalizer_fn=slim.batch_norm,
                              scope='conv1')
            feature_maps_up1 = tf.image.resize_bilinear(        # 3*3*2 -> 5*5*2
                net, [5, 5])
            feature_maps = tf.image.resize_bilinear(        # 5*5*2 -> 7*7*2
                feature_maps_up1, [7, 7])
            feature_maps_out = tf.sigmoid(feature_maps)
            excitation_bev = feature_maps_out[:,:,:,0:1]
            excitation_img = feature_maps_out[:,:,:,1:2]
            bev_feature_maps = excitation_bev * bev_feature_maps_org
            img_feature_maps = excitation_img * img_feature_maps_org

            return bev_feature_maps, img_feature_maps
