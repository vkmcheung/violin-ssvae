# -*- coding: utf-8 -*-
"""
Custom layers for proposed model in "Semi-supervised Violin Finger Generation Using Variational Autoencoders" by Vincent K.M. Cheung, Hsuan-Kai Kao, and Li Su in Proc. of the 22nd Int. Society for Music Information Retrieval Conf., Online, 2021.

"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a note."""    
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(Sampling, self).get_config()
        return dict(list(base_config.items()))  #+ list(config.items()))
    
    def call(self, inputs):
        z_mean, z_log_var = inputs  
        epsilon = tf.random.normal(shape=keras.backend.shape(z_mean), mean=0.0, stddev=1.0)
                    
        return z_mean + keras.backend.exp(z_log_var/2.)*epsilon


class KLDivergenceLayer(layers.Layer):
    """ Identity transform layer that adds KL divergence to the final model loss."""
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        kl_weight = 0.001

        mu, log_var = inputs
        kl_loss =  -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var) )
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) # take average across dimension 1 (num guassians)
        kl_loss = kl_loss*kl_weight
        self.add_loss(kl_loss, inputs=inputs)
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')   
        return inputs


#%% Gumbel Softmax for categorical sampling 2021 04 27
# Code below is modified from original code by Eric Jang: https://blog.evjang.com/2016/11/tutorial-categorical-variational.html

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0,1)"""
  U = tf.random.uniform(shape,minval=0,maxval=1)
  return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)


class GumbelSoftmaxLayer(layers.Layer):    
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(GumbelSoftmaxLayer, self).__init__(*args, **kwargs)  
    def call(self, inputs):
      """Sample from the Gumbel-Softmax distribution and optionally discretize.
      Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
      Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      """
      temperature = 0.75
      y = gumbel_softmax_sample(inputs, temperature)
      return y
        
class GumbelKLDivergenceLayer(layers.Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(GumbelKLDivergenceLayer, self).__init__(*args, **kwargs)
    def call(self, inputs):
        kl_weight = 0.001
        K = 241 #number of spf labels

        logits_y = inputs
        q_y = tf.nn.softmax(logits_y)
        log_q_y = tf.math.log(q_y+1e-20)
        
        kl_loss =  q_y*(log_q_y-tf.math.log(1.0/K))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) # take average across dimension 1 (num guassians)
        kl_loss = kl_loss*kl_weight
        self.add_loss(kl_loss, inputs=inputs)
        self.add_metric(kl_loss, name='gumbel_kl_loss', aggregation='mean')   
        return inputs    