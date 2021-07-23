# -*- coding: utf-8 -*-
"""
Implementation code for proposed model in "Semi-supervised Violin Finger Generation Using Variational Autoencoders" by Vincent K.M. Cheung, Hsuan-Kai Kao, and Li Su in Proc. of the 22nd Int. Society for Music Information Retrieval Conf., Online, 2021.

Code below is based on implementation from https://github.com/Tsung-Ping/Violin-Fingering-Generation , "Positioning Left-hand Movement in Violin Performance: A System and User Study of Fingering Pattern Generation" (IUI 2021)

"""

import numpy as np
import pickle

def load_data(data_dir):
    print('Loading data from... ' + data_dir)
    with open(data_dir, 'rb') as file:
        data = pickle.load(file)
    return data

def split_data(corpus, key_list):
    training_data = [v for k, v in corpus.items() if k in key_list]
    testing_data = [v for k, v in corpus.items() if k not in key_list]

    # make into dictionary
    training_segments = np.concatenate([x['segments'] for x in training_data], axis=0)
    training_length = [x['length'] for x in training_data]
    training_interval = np.hstack(( 0*np.ones((np.shape(training_segments)[0],1))  ,  np.diff(training_segments['pitch'], axis=1) ))
    
    testing_segments = np.concatenate([x['segments'] for x in testing_data], axis=0)
    testing_interval = np.hstack((0*np.ones((np.shape(testing_segments)[0],1)) , np.diff(testing_segments['pitch'], axis=1)))

    X = {'train': {'pitch': training_segments['pitch'],
                  'start': training_segments['start'],
                  'duration': training_segments['duration'],
                  'beat_type': training_segments['beat_type'],
                  'bow': training_segments['bow'],
                  'pure_pitch': training_segments['pure_pitch'],
                  'pure_octave': training_segments['pure_octave'],
                  'bar': training_segments['bar'],
                  'interval' : training_interval,
                  'length' : training_length
                  },
          'test':  {'pitch': testing_segments['pitch'],
                  'start': testing_segments['start'],
                  'duration': testing_segments['duration'],
                  'beat_type': testing_segments['beat_type'],
                  'bow': testing_segments['bow'],
                  'pure_pitch': testing_segments['pure_pitch'],
                  'pure_octave': testing_segments['pure_octave'],  
                  'bar': testing_segments['bar'],
                  'interval': testing_interval,
                  }}


    Y = {'train': {'string': training_segments['string']-1, #invalid labels are -1 so get assigned to last category 
                   'position': training_segments['position']-1,
                   'finger': training_segments['finger']-1,
                   },
         'test': {'string': testing_segments['string']-1,
                   'position': testing_segments['position']-1,
                   'finger': testing_segments['finger']-1,
                   }}       
    return X, Y    
