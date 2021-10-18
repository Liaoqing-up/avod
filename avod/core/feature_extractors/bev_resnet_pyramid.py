import tensorflow as tf

from avod.core.feature_extractors import bev_feature_extractor

slim = tf.contrib.slim


class BevResPyr(bev_feature_extractor.BevFeatureExtractor):
    """Contains modified res model definition to extract features from
    Bird's eye view input using pyramid features.
    """

    def res_arg_scope(self, weight_decay=0.0005):
        """Defines the resnet arg scope.

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
        with tf.variable_scope(scope, 'bev_res_pyr', [inputs]) as sc:
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
              scope='bev_res_pyr'):
        """ Modified Resnet for BEV feature extraction with pyramid features

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
            with tf.variable_scope(scope, 'bev_res_pyr', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Pad 700 to 704 to allow even divisions for max pooling
                    padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])       #704*800*6

                    block1_1 = self.bottleneck(padded, res_config.res_conv1[1], is_training, scope='block1_1')  #704*800*64
                    block1_2 = self.bottleneck(block1_1, res_config.res_conv1[1], is_training, scope='block1_2')
                    pool1 = slim.max_pool2d(block1_2, [2, 2], scope='pool1')        #352*400*64

                    block2_1 = self.bottleneck(pool1, res_config.res_conv2[1], is_training, scope='block2_1')   #352*400*128
                    block2_2 = self.bottleneck(block2_1, res_config.res_conv2[1], is_training, scope='block2_2')
                    pool2 = slim.max_pool2d(block2_2, [2, 2], scope='pool2')        #176*200*128

                    block3_1 = self.bottleneck(pool2, res_config.res_conv3[1], is_training, scope='block3_1')   #176*200*256
                    block3_2 = self.bottleneck(block3_1, res_config.res_conv3[1], is_training, scope='block3_2')
                    pool3 = slim.max_pool2d(block3_2, [2, 2], scope='pool2')        #88*100*256

                    block4_1 = self.bottleneck(pool3, res_config.res_conv4[1], is_training, scope='block4_1')   #88*100*512
                    block4_2 = self.bottleneck(block4_1, res_config.res_conv4[1], is_training, scope='block4_2')
                    # pool4 = slim.max_pool2d(block4_2, [2, 2], scope='pool2')        #44*50*512

                    # Decoder (upsample and fuse features)
                    upconv3 = slim.conv2d_transpose(        #176*200*256
                        block4_2,
                        res_config.res_conv3[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')

                    concat3 = tf.concat(
                        (block3_2, upconv3), axis=3, name='concat3')
                    pyramid_fusion3 = slim.conv2d(
                        concat3,
                        res_config.res_conv2[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3')

                    upconv2 = slim.conv2d_transpose(    #352*400*128
                        pyramid_fusion3,
                        res_config.res_conv2[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv2')

                    concat2 = tf.concat(
                        (block2_2, upconv2), axis=3, name='concat2')
                    pyramid_fusion_2 = slim.conv2d(
                        concat2,
                        res_config.res_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2')

                    upconv1 = slim.conv2d_transpose(    #704*800*64
                        pyramid_fusion_2,
                        res_config.res_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv1')

                    concat1 = tf.concat(
                        (block1_2, upconv1), axis=3, name='concat1')
                    pyramid_fusion1 = slim.conv2d(
                        concat1,
                        res_config.res_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1')

                    # Slice off padded area
                    sliced = pyramid_fusion1[:, 4:]     #700*800*64

                feature_maps_out = sliced

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points
