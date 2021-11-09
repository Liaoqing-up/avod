import tensorflow as tf

from avod.core.feature_extractors import img_feature_extractor

slim = tf.contrib.slim


class ImgResPyr(img_feature_extractor.ImgFeatureExtractor):
    """Modified Res model definition to extract features from
    RGB image input using pyramid features.
    """

    LEVEL_0 = 'level_0'
    LEVEL_1 = 'level_1'
    LEVEL_2 = 'level_2'
    LEVEL_3 = 'level_3'
    level_num = 4

    def res_arg_scope(self, weight_decay=0.0005):
        """Defines the res arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def bottleneck(self, inputs, n_out, is_training, scope=None):
        with tf.variable_scope(scope, 'img_res_pyr', [inputs]) as sc:
            conv1_1 = slim.conv2d(inputs,
                                  n_out,
                                  [3, 3],
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params={
                                      'is_training': is_training},
                                  scope='conv1'
                                  )
            conv1_2 = slim.conv2d(conv1_1,
                                  n_out,
                                  [3, 3],
                                  normalizer_fn=slim.batch_norm,
                                  activation_fn=None,
                                  normalizer_params={
                                      'is_training': is_training},
                                  scope='conv2'
                                  )
            res1 = slim.conv2d(inputs,
                               n_out,
                               [1, 1],
                               stride=1,
                               padding='SAME',
                               activation_fn=None,
                               normalizer_fn=slim.batch_norm,
                               normalizer_params={
                                   'is_training': is_training},
                               scope='res'
                               )
            bottleneck1 = tf.nn.relu(conv1_2 + res1)
            return bottleneck1

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='img_res_pyr'):
        """ Modified Resnet for image feature extraction with pyramid features.

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """

        res_config = self.config

        with slim.arg_scope(self.res_arg_scope(
                weight_decay=res_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'img_res_pyr', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    block1_1 = self.bottleneck(inputs, res_config.res_conv1[1], is_training,
                                               scope='block1_1')                # 360*1200*64
                    block1_2 = self.bottleneck(block1_1, res_config.res_conv1[1], is_training, scope='block1_2')
                    pool1 = slim.max_pool2d(block1_2, [2, 2], scope='pool1')    # 180*600*64

                    block2_1 = self.bottleneck(pool1, res_config.res_conv2[1], is_training,
                                               scope='block2_1')                # 180*600*128
                    block2_2 = self.bottleneck(block2_1, res_config.res_conv2[1], is_training, scope='block2_2')
                    pool2 = slim.max_pool2d(block2_2, [2, 2], scope='pool2')    # 90*300*128

                    block3_1 = self.bottleneck(pool2, res_config.res_conv3[1], is_training,
                                               scope='block3_1')                # 90*300*256
                    block3_2 = self.bottleneck(block3_1, res_config.res_conv3[1], is_training, scope='block3_2')
                    pool3 = slim.max_pool2d(block3_2, [2, 2], scope='pool2')    # 45*150*256

                    block4_1 = self.bottleneck(pool3, res_config.res_conv4[1], is_training,
                                               scope='block4_1')                # 45*150*512
                    block4_2 = self.bottleneck(block4_1, res_config.res_conv4[1], is_training, scope='block4_2')

                    # Decoder (upsample and fuse features)
                    tfc3 = slim.conv2d(
                        block4_2,
                        res_config.res_conv1[1],
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='transfer_channel_3')

                    tfc2 = slim.conv2d(
                        block3_2,
                        res_config.res_conv1[1],
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='transfer_channel_2')

                    tfc1 = slim.conv2d(
                        block2_2,
                        res_config.res_conv1[1],
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='transfer_channel_1')

                    tfc0 = slim.conv2d(
                        block1_2,
                        res_config.res_conv1[1],
                        [1, 1],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='transfer_channel_0')

                    upconv3 = slim.conv2d_transpose(  # 90*300*256
                        tfc3,
                        res_config.res_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')

                    add2 = tf.add(
                        tfc2, upconv3, name='add2')


                    upconv2 = slim.conv2d_transpose(  # 180*600*128
                        add2,
                        res_config.res_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv2')

                    add1 = tf.add(
                        tfc1, upconv2, name='add1')


                    upconv1 = slim.conv2d_transpose(  # 360*1200*64
                        add1,
                        res_config.res_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv1')

                    add0 = tf.add(
                        tfc0, upconv1, name='add0')

                    pyramid_level3 = slim.conv2d(
                        tfc3,
                        res_config.res_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='P3')
                    pyramid_level2 = slim.conv2d(
                        add2,
                        res_config.res_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='P2')
                    pyramid_level1 = slim.conv2d(
                        add1,
                        res_config.res_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='P1')
                    pyramid_level0 = slim.conv2d(
                        add0,
                        res_config.res_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='P0')



                # feature_maps_out = pyramid_fusion1
                feature_maps_out = dict()
                feature_maps_out[self.LEVEL_0] = pyramid_level0
                feature_maps_out[self.LEVEL_1] = pyramid_level1
                feature_maps_out[self.LEVEL_2] = pyramid_level2
                feature_maps_out[self.LEVEL_3] = pyramid_level3
                # feature_maps_out[self.LEVEL_0] = pyramid_fusion1
                # feature_maps_out[self.LEVEL_1] = pyramid_fusion_2
                # feature_maps_out[self.LEVEL_2] = pyramid_fusion_3
                # feature_maps_out[self.LEVEL_3] = block4_2


                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points

