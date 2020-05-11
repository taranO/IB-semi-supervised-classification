'''
1) to see the runned task on GPU
$ watch -n0.1 nvidia-smi

2) to run code
$ source path/to/virtual/environment/bin/activate
$ cd path/to/code
$ python3 -u test_model_BCE.py > logs.log &

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import keras
from keras.layers import Dense, Input
from keras.utils import plot_model
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from libs.constants import is_local

import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime

from libs.utils import *
from libs.mnist import MNISTData
from libs.ModelBCE import ModelBCE
from libs.ModelDc import ModelDc
from libs.ModelDz import ModelDz
from libs.ModelDxc import ModelDxc
from libs.ModelBN import ModelEncoderBN, ModelDecoderBN
from libs.ModelZC import ModelReconstructorZC, ModelEncoderZA, ModelEncoderZC

# ======================================================================================================================
print("PID = %d\n" % os.getpid())
print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ======================================================================================================================

parser = argparse.ArgumentParser(description="Classifier classical with categorical cross-entropy loss")

parser.add_argument("--classes", default=10,   type=int, help="Number of classes")
parser.add_argument("--dim_z", default=10,   type=int, help="...")
parser.add_argument("--dim_a", default=1024,   type=int, help="...")

parser.add_argument("--conv_filters", default=[32,64,128], help="")
parser.add_argument("--dense_filters", default=[1024,500], help="") # 1024,500

parser.add_argument("--is_bn",    default=False, type=int, help="...") # 2e-4

parser.add_argument("--batch_size", default=100, type=int, help="Training batch size")
parser.add_argument("--epochs",     default=100,  type=int, help="Number of training epochs")

parser.add_argument("--model_type", default="BCE_Dc",   type=str, help="Dir where to same models") # catCE, catCE_BN
parser.add_argument("--checkpoint_dir", default="model_BIB_Dc_3conv_3dense_no_bn_100_1_noise_0.30_run_1",   type=str, help="Dir where to same models") # model_E_1000   model_BN_1000

args = parser.parse_args()

# ======================================================================================================================
# Methods
def loadTestData(batch_size, data_path_):
    """ Function for load validation data """
    data = MNISTData('test',
                     data_dir=data_path_,
                     shuffle=True,
                     batch_dict_name=['im', 'label'])
    data.setup(epoch_val=0, batch_size=batch_size)
    return data

# ================================================================================================
if __name__ == "__main__":

    checkpoint_dir_ = "checkpoints/%s" % args.checkpoint_dir

    K.set_learning_phase(0)  # set learning phase: The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed as input to any Keras function that uses a different behavior at train time and test time.

    # --- print log --------------------------------------------------------
    args.dense_filters = [] if args.model_type == "model_BIB" else args.dense_filters
    info = {"checkpoint_dir_": checkpoint_dir_,
            '#_classes': args.classes, "conv_filters": args.conv_filters, 'dense_filters': args.dense_filters, "is_bn": args.is_bn,
            'batch_size': args.batch_size, 'epochs' : args.epochs, "model_type": args.model_type}
    printToLog(info)

    # ==========================================================
    # load data
    test_data = loadTestData(args.batch_size, data_path_="./data")

    unsupervised_x = Input(shape=(28, 28, 1))
    supervised_x   = Input(shape=(28, 28, 1))
    input_z = Input(shape=(args.dim_z,))
    input_c = Input(shape=(args.classes,))

    if args.model_type == "BCE_Dc":
        supervised_x = Input(shape=(28, 28, 1))

        Dc = ModelDc().init(inputs=input_c)
        Classifier = ModelBCE(num_classes=args.classes,
                              layers_conv=args.conv_filters,
                              layers_dense=args.dense_filters,
                              is_bn=args.is_bn).init(inputs=Input(shape=(28, 28, 1)))

        model = Model([supervised_x, unsupervised_x], [Classifier(supervised_x), Dc(Classifier(unsupervised_x))])

    # === Test ================================================
    Epochs = []
    Errors = []
    for epoch in range(1, args.epochs+1):
        if epoch >= 1000 and epoch % 500 != 0:
            continue
        elif epoch >= 100 and epoch % 100 != 0:
            continue
        elif epoch >= 10 and epoch % 10 != 0:
            continue

        model.load_weights("%s/model_epoch_%d" % (checkpoint_dir_, epoch))

        error = 0
        n     = 0
        cur_epoch = test_data.epochs_completed
        while cur_epoch == test_data.epochs_completed:

            batch_data = test_data.next_batch_dict()
            input = batch_data['im']
            label = batch_data['label']

            n += input.shape[0]

            y_predict = model.predict([input, input])[0]
            predict = np.argmax(y_predict, axis=1)
            diff = abs(predict - label)
            diff[diff != 0] = 1
            error += np.sum(diff)

        Epochs.append(epoch)
        Errors.append(100*error/n)

        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t epoch : {epoch}, \t"
              f"cl. error = {100*error/n}")


    ind = np.argmin(np.asarray(Errors))

    print(f"\n\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t epoch : {Epochs[ind]}, \t"
          f"cl. error = {Errors[ind]}")

































