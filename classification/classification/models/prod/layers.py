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
from classification import utility
import numpy as np

def zero_pad_features(features, depth):
    """ Zero-pad features in the depth dimension to match requested feature depth. """

    n = int(features.get_shape().dims[-1])
    extra_feature_count = depth - n
    assert n >= 0
    if n > 0:
      padding = tf.tile(features[:, :, :, :1] * 0, [1, 1, 1, extra_feature_count])
      features = tf.concat(3, [features, padding])
    return features


def feature_pool(layer, pool_sz, pool_fn=tf.reduce_max):
    H, W, D = [int(x) for x in layer.get_shape().dims[1:]]
    assert D % pool_sz == 0, "pool_sz ({}) must evenly divide D ({})".format(pool_sz, D)
    layer = tf.reshape(layer, [-1, H, W, D // pool_sz, pool_sz])
    return pool_fn(layer, -1)



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


def misconception_layer2(input,
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
                activation_fn=tf.nn.relu,
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
                              count=1,
                              scope=None):
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        misconception = input
        for _ in range(count-1):
          misconception = misconception_layer(misconception, window_size, 1, depth,
                                              is_training, scope)
        misconception = misconception_layer(misconception, window_size, stride, depth,
                                            is_training, scope)
        bypass = slim.avg_pool2d(
            input, [1, window_size], stride=[1, stride], padding='SAME')

        n = int(bypass.get_shape().dims[-1])
        if n != depth:
          bypass = slim.conv2d(
                bypass, depth, [1, 1], stride=[1, 1], activation_fn=None)


        return misconception + bypass


def misconception_with_bypass2(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              count=1,
                              scope=None):
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        misconception = input
        for _ in range(count-1):
          misconception = misconception_layer(misconception, window_size, 1, depth,
                                              is_training, scope)
        misconception = misconception_layer(misconception, window_size, stride, depth,
                                            is_training, scope)


        bypass = input
        if stride > 1:
          bypass = slim.avg_pool2d(
                        bypass, [1, stride], stride=[1, stride], padding='SAME')
        bypass = zero_pad_features(bypass, depth)

        return misconception + bypass


def misconception_with_bypass3(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              scope=None):
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
          residual = misconception_layer2(misconception, window_size, stride, depth,
                                            is_training, scope)

          if stride > 1:
              input = slim.avg_pool2d(
                          input, [1, stride], stride=[1, stride], padding='SAME')

          input = zero_pad_features(input, depth)

          return input + residual

def deception_layer(input,
                        window_size,
                        stride,
                        depth,
                        count,
                        is_training,
                        gate_hidden=32,
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
            stage_max_pool_reduce = slim.max_pool2d(
                input, [1, window_size], stride=[1, stride], padding='SAME')
            stage_max_pool_reduce = slim.conv2d(stage_max_pool_reduce, depth, [1, 1])
            layers = [tf.expand_dims(stage_max_pool_reduce, 4)]
            for _ in range(count):
              stage_conv = slim.conv2d(
                  input, depth, [1, window_size], stride=[1, stride])
              layers.append(tf.expand_dims(stage_conv, 4))

            layers = tf.concat(4, layers)

            gate = slim.conv2d( # Try this instead
                  input, gate_hidden, [1, window_size], stride=[1, stride], activation_fn=tf.nn.elu)
            gate = slim.conv2d(
                  gate, (count + 1), [1, window_size], stride=[1, 1], activation_fn=tf.nn.softmax)

            gate = tf.expand_dims(gate, 3)

            # `gate` is Bx1xWx1xL
            # `layers` is Bx1xWxDxL
            # multiply broadcasts to Bx1xWxDxL, then reduce across layers
            results = tf.reduce_sum(gate * layers, 4)
            # results = stage_max_pool_reduce * gate[:, :, :, count:]
            # for i in range(count):
            #   results = results + layers[i] * gate[:, :, :, i:i+1]

            return results


def deception_with_bypass(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              sub_levels=1,
                              count=3,
                              scope=None):
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        deception = input
        for _ in range(sub_levels-1):
          deception = deception_layer(deception, window_size, 1, depth, count,
                                              is_training, gate_hidden=32, scope=scope)
        deception = deception_layer(deception, window_size, stride, depth, count,
                                            is_training, gate_hidden=32, scope=scope)
        bypass = slim.avg_pool2d(
            input, [1, window_size], stride=[1, stride], padding='SAME')

        n = int(bypass.get_shape().dims[-1])
        if n != depth:
          bypass = slim.conv2d(
                bypass, depth, [1, 1], stride=[1, 1], activation_fn=None)


        return deception + bypass



def basic_with_bypass(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              sub_levels=1,
                              scope=None):
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=tf.nn.elu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
          basic = input
          for i in range(sub_levels):
              basic = slim.conv2d(basic, depth, [1, window_size], stride=[1, 1], padding='SAME')

          bypass = input

          if stride > 1:
            basic = slim.max_pool2d(
                  basic, [1, stride], stride=[1, stride], padding='SAME')

            bypass = slim.avg_pool2d(
                        input, [1, stride], stride=[1, stride], padding='SAME')

          n = int(bypass.get_shape().dims[-1])
          if n != depth:
            bypass = slim.conv2d(
                  bypass, depth, [1, 1], stride=[1, 1], activation_fn=None)

          return basic + bypass


def basic_with_bypass2(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              sub_levels=1,
                              scope=None):
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=tf.nn.elu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
          basic = input
          for i in range(sub_levels):
              basic = slim.conv2d(basic, depth, [1, window_size], stride=[1, 1], padding='SAME')

          basic = slim.max_pool2d(
                basic, [1, window_size], stride=[1, stride], padding='SAME')

          bypass = slim.avg_pool2d(
                        input, [1, window_size], stride=[1, stride], padding='SAME')
          bypass = zero_pad_features(bypass, depth)

          return basic + bypass


def basic_with_bypass3(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              scope=None):
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=tf.nn.elu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
          residual = slim.conv2d(input, depth, [1, window_size], stride=[1, 1], padding='SAME')

          if stride > 1:
              residual = slim.max_pool2d(
                residual, [1, stride], stride=[1, stride], padding='SAME')

              input = slim.avg_pool2d(
                          input, [1, stride], stride=[1, stride], padding='SAME')

          input = zero_pad_features(input, depth)

          return input + residual


def maxout_with_bypass(input,
                        window_size,
                        stride,
                        depth,
                        is_training,
                        maxout_sz=2,
                        scope=None):
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=None,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
          residual = slim.conv2d(input, maxout_sz * depth, [1, window_size], stride=[1, stride], padding='SAME')
          residual = feature_pool(residual, maxout_sz)

          if stride > 1:
            input = slim.avg_pool2d(
                          input, [1, stride], stride=[1, stride], padding='SAME')
          input = zero_pad_features(input, depth)

          return input + residual


def inception_layer(input,
                    window_size,
                    stride,
                    depth,
                    is_training,
                    scope=None):
    """ A single layer of the  convolutional network.

  Args:
    input: a tensor of size [batch_size, 1, width, depth]
    window_size: the width of the conv and pooling filters to apply.
    stride: the downsampling to apply when filtering.
    depth: the depth of the output tensor.

  Returns:
    a tensor of size [batch_size, 1, width/stride, depth].
  """
    assert depth % 2 == 0, 'depth must be divisible by 2'
    sub_depth = depth // 2
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=tf.nn.elu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
            stage_conv = slim.conv2d(input, sub_depth, [1, 1])
            stage_conv = slim.conv2d(
                stage_conv, sub_depth, [1, window_size], stride=[1, stride])
            stage_pool = slim.max_pool2d(
                input, [1, window_size], stride=[1, stride], padding='SAME')
            stage_pool = slim.conv2d(stage_pool, sub_depth, [1, 1])

            return tf.concat(3, [stage_conv, stage_pool])



def inception_with_bypass(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              count=1,
                              scope=None):
    """ A inception added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        inception = input
        for _ in range(count-1):
          inception = inception_layer(inception, window_size, 1, depth,
                                              is_training, scope)
        inception = inception_layer(inception, window_size, stride, depth,
                                            is_training, scope)
        bypass = slim.avg_pool2d(
            input, [1, window_size], stride=[1, stride], padding='SAME')

        return inception + bypass




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


    with slim.arg_scope([slim.batch_norm], decay=0.999):
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




def xception_with_bypass(input,
                        window_size,
                        depth,
                        is_training,
                        count=1,
                        scope=None):
    with tf.name_scope(scope):
        n = int(input.get_shape().dims[3])
        net = input
        with slim.arg_scope(
            [slim.separable_conv2d],
                padding='SAME',
                activation_fn=None,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
          for _ in range(count):
            net = tf.nn.elu(net)
            net = slim.separable_conv2d(net, num_outputs=depth, kernel_size=[1, window_size], depth_multiplier=1)
          net = slim.max_pool2d(net, [1, 3], stride=[1, 2], padding='SAME')
          bypass = slim.avg_pool2d(input, [1, 2], stride=[1, 2], padding='SAME')
          if n != depth:
            bypass = slim.conv2d(bypass, depth, [1, 1], stride=[1, 1],
                                 normalizer_fn=slim.batch_norm,
                                  normalizer_params={'is_training': is_training})
          net = bypass + net
    return net


def xception_with_bypass2(input,
                        window_size,
                        depth,
                        is_training,
                        count=1,
                        scope=None):

    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.separable_conv2d],
                padding='SAME',
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
          residual = slim.separable_conv2d(input, num_outputs=depth, 
                kernel_size=[1, window_size], stride=[1, 1], depth_multiplier=1, padding='SAME')

          if stride > 1:
              residual = slim.max_pool2d(
                residual, [1, stride], stride=[1, stride], padding='SAME')

              input = slim.avg_pool2d(
                          input, [1, stride], stride=[1, stride], padding='SAME')

          input = zero_pad_features(input, depth)

          return input + residual




def xception_model(input, window_size, depth, levels,
                        objective_functions, is_training, dense_count=100, dense_layers=1, 
                        sub_levels=3, final_keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    with slim.arg_scope([slim.batch_norm], decay=0.999):
      with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
          net = slim.conv2d(
                  input, depth, [1, window_size],
                  padding='SAME',
                  activation_fn=None,
                  normalizer_fn=slim.batch_norm,
                  normalizer_params={'is_training': is_training})          
          net = slim.repeat(net, levels, xception_with_bypass, window_size, depth, is_training,
            count=sub_levels)
          net = tf.nn.elu(net)
          net = slim.flatten(net)
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



def xception_model2(input, window_size, depth, levels,
                        objective_functions, is_training, dense_count=100, dense_layers=1, 
                        sub_levels=3, final_keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    with slim.arg_scope([slim.batch_norm], decay=0.999):
      with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
          net = slim.conv2d(
                  input, depth, [1, window_size],
                  padding='SAME',
                  activation_fn=None,
                  normalizer_fn=slim.batch_norm,
                  normalizer_params={'is_training': is_training})          
          net = slim.repeat(net, levels, xception_with_bypass, window_size, depth, is_training,
            count=sub_levels)
          net = tf.nn.elu(net)
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



def xception_model3(input, window_size, depth, levels,
                        objective_functions, is_training, dense_count=1024, dense_layers=2, 
                        sub_levels=3, keep_prob=0.5,
                        l2=1e-5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected], 
        weights_regularizer=slim.l2_regularizer(l2)):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
              net = slim.conv2d(
                      input, depth, [1, window_size],
                      stride=[1, 2],
                      padding='SAME',
                      activation_fn=None,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': is_training})          
              net = slim.repeat(net, levels, xception_with_bypass, window_size, depth, is_training,
                count=sub_levels)
              net = tf.nn.elu(net)
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

              return outputs



def xception_model4(input, window_size, depth, levels,
                        objective_functions, is_training, depth_inc=0, dense_count=256, dense_layers=2, 
                        sub_levels=3, keep_prob=0.5,
                        l2=1e-5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = [input]
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected], 
        weights_regularizer=slim.l2_regularizer(l2)):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
              net = slim.conv2d(
                      input, depth, [1, window_size],
                      stride=[1, 2],
                      padding='SAME',
                      activation_fn=None,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': is_training})        
              layers.append(net)  
              for _ in range(levels):
                net = xception_with_bypass(net, window_size, depth, is_training, count=sub_levels)
                layers.append(net)
                depth += depth_inc
              net = tf.nn.elu(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              # Flatten
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers



def xception_model5(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = misconception_with_bypass2(net, window_size, stride, depth, is_training)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers



def misconception_model2(input, window_size, depth, levels,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
              net = slim.conv2d(
                      input, depth, [1, 1],
                      stride=[1, 1],
                      padding='SAME',
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': is_training})        
              layers.append(net)  
              for _ in range(levels):
                net = misconception_with_bypass(net, window_size, 2, depth, is_training, count=sub_levels)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              # Flatten
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers



def misconception_model3(input, window_size, depth, levels,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
              net = slim.conv2d(
                      input, depth, [1, 1],
                      stride=[1, 1],
                      padding='SAME',
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': is_training})        
              layers.append(net)  
              for _ in range(levels):
                net = misconception_with_bypass(net, window_size, 2, depth, is_training, count=sub_levels)
                layers.append(net)
              # Flatten
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers



def misconception_model4(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = misconception_with_bypass(net, window_size, stride, depth, is_training, count=sub_levels)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              # Flatten
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers


def misconception_model5(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = misconception_with_bypass(net, window_size, stride, depth, is_training, count=sub_levels)
                layers.append(net)
              # Flatten
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers



def misconception_model6(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = misconception_with_bypass(net, window_size, stride, depth, is_training, count=sub_levels)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers

def misconception_model7(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = misconception_with_bypass(net, window_size, stride, depth, is_training, count=sub_levels)
                layers.append(net)
              # Flatten
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers


def misconception_model7(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = misconception_with_bypass2(net, window_size, stride, depth, is_training)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers




def misconception_model8(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.batch_norm], decay=0.999):
      with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):  
          net = input    
          layers.append(net)  
          for depth, stride in zip(depths, strides):
            net = misconception_with_bypass2(net, window_size, stride, depth, is_training)
            layers.append(net)
          # Global average pool
          n = int(net.get_shape().dims[1])
          net = slim.avg_pool2d(net, [1, n], stride=[1,n])
          net = slim.flatten(net)
          outputs = []
          for ofunc in objective_functions:
            onet = net
            for _ in range(dense_layers - 1):
              onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                         normalizer_params={'is_training': is_training})
            # Don't use batch norm on last layer, just use dropout.
            onet = slim.fully_connected(onet, dense_count)
            onet = slim.dropout(onet, keep_prob, is_training=is_training)
            outputs.append(ofunc.build(onet))

    return outputs, layers



def deception_model(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5, count=3):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = deception_with_bypass(net, window_size, stride, depth, is_training, count=count, sub_levels=sub_levels)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers


def basic_model(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = basic_with_bypass(net, window_size, stride, depth, is_training, sub_levels=sub_levels)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers


def basic_model2(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = basic_with_bypass3(net, window_size, stride, depth, is_training)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers


def maxout_model(input, window_size, depths, strides,
                        objective_functions, is_training, dense_count=256, dense_layers=2, 
                        keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):  
              net = input    
              layers.append(net)  
              for depth, stride in zip(depths, strides):
                net = maxout_with_bypass(net, window_size, stride, depth, is_training)
                layers.append(net)
              # Global average pool
              n = int(net.get_shape().dims[1])
              net = slim.avg_pool2d(net, [1, n], stride=[1,n])
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers



def inception_model(input, window_size, depth, levels,
                        objective_functions, is_training, dense_count=1024, dense_layers=2, 
                        sub_levels=1, keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = [input]
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected]):
        with slim.arg_scope([slim.batch_norm], decay=0.999):
          with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
              net = slim.conv2d(
                      input, depth, [1, 1],
                      stride=[1, 1],
                      padding='SAME',
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': is_training})        
              layers.append(net)                
              for _ in range(levels):
                net = inception_with_bypass(net, window_size, 2, depth, is_training, count=sub_levels)
                layers.append(net)
              # Flatten
              net = slim.flatten(net)
              outputs = []
              for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                  onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
                                                             normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers



# def misconception_model3(input, window_size, depth, levels,
#                         objective_functions, is_training, dense_count=256, dense_layers=2, 
#                         sub_levels=1, keep_prob=0.5,
#                         l2=1e-6):
#     """ A misconception tower.

#   Args:
#     input: a tensor of size [batch_size, 1, width, depth].
#     window_size: the width of the conv and pooling filters to apply.
#     depth: the depth of the output tensor.
#     levels: the height of the tower in misconception layers.
#     objective_functions: a list of objective functions to add to the top of
#                          the network.
#     is_training: whether the network is training.

#   Returns:
#     a tensor of size [batch_size, num_classes].
#   """
#     layers = [input]
#     with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected], 
#         weights_regularizer=slim.l2_regularizer(l2)):
#         with slim.arg_scope([slim.batch_norm], decay=0.999):
#           with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
#               net = slim.conv2d(
#                       input, depth, [1, 1],
#                       stride=[1, 1],
#                       padding='SAME',
#                       activation_fn=None,
#                       normalizer_fn=slim.batch_norm,
#                       normalizer_params={'is_training': is_training}) 
#               net = tf.nn.elu(net)       
#               layers.append(net)  
#               for _ in range(levels):
#                 net = misconception_with_bypass(net, window_size, 2, depth, is_training, count=sub_levels)
#                 layers.append(net)
#               # Global average pool
#               n = int(net.get_shape().dims[1])
#               net = slim.avg_pool2d(net, [1, n], stride=[1,n])
#               # Flatten
#               net = slim.flatten(net)
#               outputs = []
#               for ofunc in objective_functions:
#                 onet = net
#                 for _ in range(dense_layers - 1):
#                   onet = slim.fully_connected(onet, dense_count, normalizer_fn=slim.batch_norm, 
#                                                              normalizer_params={'is_training': is_training})
#                 # Don't use batch norm on last layer, just use dropout.
#                 onet = slim.fully_connected(onet, dense_count)
#                 onet = slim.dropout(onet, keep_prob, is_training=is_training)
#                 outputs.append(ofunc.build(onet))

#     return outputs, layers



def misconception_fishing(input, window_size, depth, levels,
                        objective_function, is_training, dense_count=16, dense_layers=2, 
                        count=1,
                        keep_prob=0.5,
                        l2=1e-6):

  _, layers = misconception_model2(input, window_size, depth, levels,
                        [], is_training, dense_count=dense_count, dense_layers=2, 
                        keep_prob=keep_prob,
                        l2=l2,
                        count=count)

  # First layer is raw input and we don't want to apply activation to that.
  # Other layers don't get activation functions in xception_model
  expanded_layers = [layers[0]]
  for i, lyr in enumerate(layers):
    if i > 0:
      expanded_layers.append(tf.nn.elu(utility.repeat_tensor(lyr, 2**i)))

  embedding = tf.concat(3, expanded_layers)

  for _ in range(dense_layers-1):
    embedding = slim.conv2d(embedding, dense_count, [1, 1],
          activation_fn=tf.nn.elu, 
          normalizer_fn=slim.batch_norm,
          normalizer_params={'is_training': is_training})
  embedding = slim.conv2d(embedding, dense_count, [1, 1],
                          activation_fn=tf.nn.elu,
                          normalizer_fn=None)
  embedding = slim.dropout(embedding, keep_prob, is_training=is_training)

  fishing_outputs = tf.squeeze(
      slim.conv2d(
          embedding,
          1, [1, 1],
          activation_fn=None,
          normalizer_fn=None),
      squeeze_dims=[1, 3])

  return objective_function.build(fishing_outputs)


def misconception_fishing5(input, window_size, depths, strides,
                        objective_function, is_training, dense_count=16, dense_layers=2, 
                        keep_prob=0.5,
                        l2=1e-6):

  _, layers = misconception_model5(input, window_size, depths, strides,
                        [], is_training, dense_count=dense_count, dense_layers=2, 
                        keep_prob=keep_prob)

  # First layer is raw input and we don't want to apply activation to that.
  # Other layers don't get activation functions in xception_model
  expanded_layers = [layers[0]]
  for i, lyr in enumerate(layers):
    if i > 0:
      expanded_layers.append(tf.nn.elu(utility.repeat_tensor(lyr, 2**i)))

  embedding = tf.concat(3, expanded_layers)

  for _ in range(dense_layers-1):
    embedding = slim.conv2d(embedding, dense_count, [1, 1],
          activation_fn=tf.nn.elu, 
          normalizer_fn=slim.batch_norm,
          normalizer_params={'is_training': is_training})
  embedding = slim.conv2d(embedding, dense_count, [1, 1],
                          activation_fn=tf.nn.elu,
                          normalizer_fn=None)
  embedding = slim.dropout(embedding, keep_prob, is_training=is_training)

  fishing_outputs = tf.squeeze(
      slim.conv2d(
          embedding,
          1, [1, 1],
          activation_fn=None,
          normalizer_fn=None),
      squeeze_dims=[1, 3])

  return objective_function.build(fishing_outputs)



def misconception_fishing6(input, window_size, depths, strides,
                        objective_function, is_training, dense_count=16, dense_layers=2, 
                        keep_prob=0.5,
                        l2=1e-6):

  _, layers = misconception_model6(input, window_size, depths, strides,
                        [], is_training, dense_count=dense_count, dense_layers=2, 
                        keep_prob=keep_prob)

  # First layer is raw input and we don't want to apply activation to that.
  # Other layers don't get activation functions in xception_model
  expanded_layers = [layers[0]]
  for i, lyr in enumerate(layers):
    if i > 0:
      expanded_layers.append(tf.nn.elu(utility.repeat_tensor(lyr, 2**i)))

  # Add an edge distance feature that indicates how far we are from edge of
  # the model.
  n = int(input.get_shape().dims[2])
  edge_dist = np.minimum(np.arange(n), np.arange(n)[::-1], dtype='float32')
  edge_dist /= (n // 2) 
  edge_dist = edge_dist.reshape([1, 1, n, 1])
  edge_dist = tf.tile(tf.Variable(edge_dist), [tf.shape(input)[0], 1, 1, 1])

  expanded_layers.append(edge_dist)

  embedding = tf.concat(3, expanded_layers)

  for _ in range(dense_layers-1):
    embedding = slim.conv2d(embedding, dense_count, [1, 1],
          activation_fn=tf.nn.elu, 
          normalizer_fn=slim.batch_norm,
          normalizer_params={'is_training': is_training})
  embedding = slim.conv2d(embedding, dense_count, [1, 1],
                          activation_fn=tf.nn.elu,
                          normalizer_fn=None)
  embedding = slim.dropout(embedding, keep_prob, is_training=is_training)

  fishing_outputs = tf.squeeze(
      slim.conv2d(
          embedding,
          1, [1, 1],
          activation_fn=None,
          normalizer_fn=None),
      squeeze_dims=[1, 3])

  return objective_function.build(fishing_outputs)
 