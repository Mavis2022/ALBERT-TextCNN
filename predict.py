# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:12:37 2019

@author: cm
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import tensorflow as tf
from networks import NetworkAlbertTextCNN
from classifier_utils import get_feature_test,id2label
from hyperparameters import Hyperparamters as hp
import pandas as pd
#import sagemaker
import pickle
#import sagemaker
          

class ModelAlbertTextCNN(object,):
    """
    Load NetworkAlbert TextCNN model
    """
    def __init__(self):
        self.albert, self.sess = self.load_model()
    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert =  NetworkAlbertTextCNN(is_training=False)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(pwd,hp.file_save_model))
                print (checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert,sess

MODEL = ModelAlbertTextCNN()
print('Load model finished!')


def get_label(sentence):
    """
    Prediction of the sentence's label.
    """
#     feature = get_feature_test(sentence)
#     fd = {MODEL.albert.input_ids: [feature[0]],
#           MODEL.albert.input_masks: [feature[1]],
#           MODEL.albert.segment_ids:[feature[2]],
#           }
#     prediction = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)[0]
#     return [id2label(l) for l in np.where(prediction==1)[0] if l!=0]
    feature = get_feature_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
    MODEL.albert.input_masks: [feature[1]],
    MODEL.albert.segment_ids:[feature[2]],
    }
    prediction = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)[0]
#    print(prediction)
    r=[]
    # single-label
    return [id2label(prediction)]
    # multi-label
#    for i in range(len(prediction)):
#        if prediction[i]!=0.0:
#            r.append(id2label(i))
#    return r
#    # return [id2label(l) for l in np.where(prediction==1)[0] if l!=0]

#def upload_s3(local_path, key_prefix):
#    return sess.upload_data(path=local_path, bucket='bigonelab-machine-learning', key_prefix=key_prefix)

if __name__ == '__main__':
    # Test
#     sentences = ['耗电情况：整体来说耗电不是特别严重',
#      '取暖效果：取暖效果好',
#      '取暖效果：开到二挡很暖和',
#      '一个小时房间仍然没暖和',
#      '开着坐旁边才能暖和']
#    sess = sagemaker.Session()
#    role = sagemaker.get_execution_role()
    sentences = []
    with open('data/golden_set_to_pred_v2.txt','r') as f:
        for line in f:
            sentences.append(line)
    print(len(sentences))
    #f = open('data/predict_golden_set_result_v2.txt','a')
    df = pd.DataFrame()
    i = 1
    for sentence in sentences:
        sentence=sentence.strip('\n')
        print(i,sentence,get_label(sentence))
        #f.write(str([i,sentence,get_label(sentence)]))  # 将预测结果写入文件中
        #f.write("\n")  # 换行
        df2 = pd.DataFrame([[i,sentence,get_label(sentence)]],columns=['index','spu_name','label'])
        df = pd.concat([df, df2])
        print(df)
        df.to_csv('data/predict_golden_set_result_0707_v1.csv')
        i = i+1
#    upload_s3(f1, 'nlp/Wenxue/consumer_goods')


    
    
    
    


