from classification.model import ModelBase, TrainNetInfo

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model(ModelBase):

    def build_network(self, features):

        net = slim.repeat(slim.flatten(features), 3, slim.fully_connected,
            100, activation_fn=tf.nn.relu)
        return slim.fully_connected(net, self.num_classes)

    def build_training_net(self, features, labels):

        one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)

        logits = self.build_network(features)
        loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        optimizer = tf.train.AdamOptimizer()

        return TrainNetInfo(loss, optimizer, logits)

    def build_inference_net(self, features):

        return self.build_network(features)
