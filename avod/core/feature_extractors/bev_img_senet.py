"""Contains modified SENet model  to apply attention mechanism to bev and img feature_map.

Usage:
    outputs, end_points = BevVgg(inputs, layers_config)
"""

import tensorflow as tf


slim = tf.contrib.slim


class SeModule():
    def __init__(self, ratio):
        self.ratio = ratio  # 4

    def build(self, bev_feature_maps_org, img_feature_maps_org):
        with tf.variable_scope('bev_img_senet'):
            hb, wb, cb = tuple([dim.value for dim in bev_feature_maps_org.shape[1:4]])
            hi, wi, ci = tuple([dim.value for dim in img_feature_maps_org.shape[1:4]])
            c = int(cb) + int(ci)     # 64+64=128 # 32+32=64

            bev_feature_line_avg = slim.avg_pool2d(bev_feature_maps_org, [hb, wb], padding='valid')     # 64
            img_feature_line_avg = slim.avg_pool2d(img_feature_maps_org, [hi, wi], padding='valid')     # 64

            bev_feature_line_max = slim.max_pool2d(bev_feature_maps_org, [hb, wb], padding='valid')     # 64
            img_feature_line_max = slim.max_pool2d(img_feature_maps_org, [hi, wi], padding='valid')     # 64
            excitation_bev_max = slim.flatten(bev_feature_line_max)
            excitation_img_max = slim.flatten(img_feature_line_max)

            # bev_feature_line = slim.flatten(bev_feature_maps_org, scope='flatten_bev')
            # img_feature_line = slim.flatten(img_feature_maps_org, scope='flatten_img')

            excitation_bev_avg = slim.flatten(bev_feature_line_avg)
            excitation_img_avg = slim.flatten(img_feature_line_avg)
            excitation = tf.concat(                                 # # (1, 64*4)
                [excitation_bev_avg, excitation_bev_max, excitation_img_avg, excitation_img_max], axis=-1)
            # print("contact excitation shape:", excitation.shape)
            excitation = slim.fully_connected(excitation, int(c/self.ratio), scope='se_fc1',    # (1, 32) # (1, 16)
                                              weights_regularizer=None,
                                              weights_initializer=slim.xavier_initializer(),
                                              activation_fn=tf.nn.relu)
            # print("se_fc1 excitation shape:", excitation.shape)
            excitation = slim.fully_connected(excitation, c, scope='se_fc2',                    # (1, 128) # (1, 64)
                                              weights_regularizer=None,
                                              weights_initializer=slim.xavier_initializer(),
                                              activation_fn=tf.nn.sigmoid)
            # weight = slim.softmax(excitation, scope='softmax')
            # print("se_fc2 excitation shape:", excitation.shape)
            se_weight = tf.reshape(excitation, [-1, 1, 1, c])               # (1, 1, 1, 128) # (1, 1, 1, 64)
            # print("se_weight shape:", se_weight.shape)
            bev_weight = se_weight[:,:,:,0:cb]                              # (1, 1, 1, 64) # (1, 1, 1, 32)
            # print("bev_weight shape:", bev_weight.shape)
            img_weight = se_weight[:,:,:,cb:]                               # (1, 1, 1, 64) # (1, 1, 1, 32)
            # print("img_weight shape:", img_weight.shape)
            bev_feature_maps = bev_feature_maps_org * bev_weight            # (1, 700, 800, 64) # (1, 700, 800, 32)
            # print("bev_feature_maps shape:", bev_feature_maps.shape)
            img_feature_maps = img_feature_maps_org * img_weight            # (1, 700, 800, 64) # (1, 360, 1200, 32)
            # print("img_feature_maps shape:", img_feature_maps.shape)

            return bev_feature_maps, img_feature_maps
