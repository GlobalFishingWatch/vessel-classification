# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow.contrib.slim as slim


def misconception_layer(input,
                        window_size,
                        stride,
                        depth,
                        is_training,
                        scope=None):
    """ A single layer of the misconception convolutional network.

  Args:
    input: a tensor of size [batch_size, 1, width, depth]
    window_size: the width of the conv and pooling filters to apply.
    stride: the downsampling to apply when filtering.
    depth: the depth of the output tensor.

  Returns:
    a tensor of size [batch_size, 1, width/stride, depth].
  """
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=tf.nn.elu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
            stage_conv = slim.conv2d(
                input, depth, [1, window_size], stride=[1, stride])
            stage_max_pool_reduce = slim.max_pool2d(
                input, [1, window_size], stride=[1, stride], padding='SAME')

            concat = tf.concat(3, [stage_conv, stage_max_pool_reduce])

            return slim.conv2d(concat, depth, [1, 1])


def misconception_with_bypass(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              scope=None):
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        misconception = misconception_layer(input, window_size, stride, depth,
                                            is_training, scope)
        bypass = slim.avg_pool2d(
            input, [1, window_size], stride=[1, stride], padding='SAME')

        return misconception + bypass


def misconception_model(input, window_size, stride, depth, levels,
                        objective_functions, is_training, dense_count=100, dense_layers=1, final_keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    stride: the downsampling to apply when filtering.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    with slim.arg_scope([slim.batch_norm], decay=0.9999):
      with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
          net = input
          net = slim.repeat(net, levels, misconception_with_bypass, window_size,
                            stride, depth, is_training)
          net = slim.flatten(net)
          net = slim.dropout(net, final_keep_prob, is_training=is_training)
          outputs = []
          for ofunc in objective_functions:
            onet = net
            for _ in range(dense_layers - 1):
              onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                         normalizer_params={'is_training': is_training})
            # Don't use batch norm on last layer, just use dropout.
            onet = slim.fully_connected(onet, dense_count)
            onet = slim.dropout(onet, final_keep_prob, is_training=is_training)
            outputs.append(ofunc.build(onet))

          return outputs
