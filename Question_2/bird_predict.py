# TIPP - AAI Assignment (Applied AI Solutions Development – Computer Vision)
# Date Due: 6 March 2020
# Submited By: KOAY SENG TIAN
# Email: sengtian@yahoo.com
#
# GitHub: https://github.com/koayst/rp_computer_vision_assignment
#
# Instruction:
#    python bird_predict.py -t test_bird -m ..\model\bird_cv_model.h5 -e ..\model\bird_cv_labels.pkl
#
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--testdir', required=True, help='path to test directory')
ap.add_argument('-m', '--model', required=True, help='model file name')
ap.add_argument('-e', '--encoder', required=True, help='encoder file name')
args = vars(ap.parse_args())

from imutils import paths
from keras.models import load_model
from keras.preprocessing import image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf

script_dir = os.getcwd()
test_dir = args['testdir']
model_name = args['model']
label_name = args['encoder']

# print('Current directory: {}'.format(script_dir))
# print('Test data directory: {}'.format(args['testdir']))
# print('Model file name: {}'.format(args['model']))
# print('Encoder file name: {}'.format(args['encoder']))

# to hide tensorflow warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

# load the trained model
model = load_model(os.path.join(script_dir, '..\\model\\' + model_name))
#model.summary()
print('Load model.')

# load the LabelEncoder
with open(os.path.join(script_dir, label_name), 'rb+') as encoder_file:
    lb = pickle.load(encoder_file)
    print('Load LabelEncoder.')

lb_name_mapping = dict(zip(lb.transform(lb.classes_), lb.classes_))

# image size required by VGG16 is 224 x 224
img_size = 224

print()
for imgFile in sorted(list(paths.list_images(test_dir))):
    image = cv2.imread(imgFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))

    data = np.array(image)
    print('Original:', imgFile.split(os.path.sep)[-1])
    
    data = np.expand_dims(data, axis=0)

    preds = model.predict(data)
    top_three_prob = np.sort(preds[0])[-1:-4:-1]
    top_three_indices = np.argsort(preds[0])[-1:-4:-1]

    for i in range(len(top_three_prob)):
        print('{:02d}: {:{}{}{}} |{:05.2f}%|'.format(
        	i + 1, 
        	lb_name_mapping[top_three_indices[i]], '.', '<', 30, top_three_prob[i] * 100))
              
    print()
