'''
Variational Information Bottleneck for Semi-supervised Classification
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras import backend as K
from keras.models import Model

import argparse
import datetime

from libs.utils import *
from libs.mnist import MNISTData
from libs.ModelDc import ModelDc
from libs.ModelDz import ModelDz
from libs.ModelBN import ModelEncoderBN, ModelDecoderBN
from libs.ModelDcc import ModelDcc
from libs.ModelZC import ModelEncoderZC

# ======================================================================================================================
print("PID = %d\n" % os.getpid())
print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================================================================================================================

parser = argparse.ArgumentParser(description="Variational Information Bottleneck for Semi-supervised Classification")

parser.add_argument("--classes", default=10,   type=int, help="Number of classes")

parser.add_argument("--conv_filters",  default=[32,64,128], help="Encoder's convolutional layers")
parser.add_argument("--dense_filters", default=[], help="Encoder's dense layers")
parser.add_argument("--dim_z",   default=10,   type=int, help="The dimensionality of latent space a")
parser.add_argument("--dim_a",   default=1024, type=int, help="The dimensionality of latent space z")
parser.add_argument("--is_bn",    default=False, type=int, help="Is to apply batch normalisation in the encoder")

parser.add_argument("--batch_size", default=100, type=int, help="Training batch size")
parser.add_argument("--epochs",     default=100,  type=int, help="Number of training epochs")

# Trained model types: Dcc_Dc, Dcc_Da, Dcc_Da_Dc, Dcc_Dz_Dc_Dxx, Dcc_Dz_Dc_Dxx_Dx
parser.add_argument("--model_type", default="Dcc_Dc",   type=str, help="Trained model type")
parser.add_argument("--checkpoint_dir", default="Dcc_Dc/...",   type=str, help="Path to the saved models")

args = parser.parse_args()

# ======================================================================================================================
# Methods
def loadTestData(batch_size, data_path_):
    """ Function for load validation data """
    data = MNISTData('test',
                     data_dir=data_path_,
                     shuffle=False,
                     batch_dict_name=['im', 'label'])
    data.setup(epoch_val=0, batch_size=batch_size)
    return data

# ======================================================================================================================
if __name__ == "__main__":

    checkpoint_dir_ = "checkpoints/%s" % args.checkpoint_dir

    K.set_learning_phase(0)  # set learning phase: The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed as input to any Keras function that uses a different behavior at train time and test time.

    # --- print log --------------------------------------------------------
    args.dense_filters = [] if args.model_type == "model_BIB" else args.dense_filters
    info = {"checkpoint_dir_": checkpoint_dir_,
            '#_classes': args.classes, "conv_filters": args.conv_filters, 'dense_filters': args.dense_filters, "is_bn": args.is_bn,
            'batch_size': args.batch_size, 'epochs' : args.epochs, "model_type": args.model_type}
    printToLog(info)

    # === load test data ====================================================================================================
    test_data = loadTestData(args.batch_size, data_path_="./data")

    # --- init models -------------------------------------------------------
    unsupervised_x = Input(shape=(28, 28, 1))
    supervised_x   = Input(shape=(28, 28, 1))
    input_z = Input(shape=(args.dim_z,))
    input_c = Input(shape=(args.classes,))

    if args.model_type == "Dcc_Dc":
        Dc = ModelDc().init(inputs=input_c)
        Classifier = ModelDcc(num_classes=args.classes,
                              layers_conv=args.conv_filters,
                              layers_dense=args.dense_filters,
                              is_bn=args.is_bn).init(inputs=Input(shape=(28, 28, 1)))

        model = Model(unsupervised_x, [Classifier(unsupervised_x),
                                       Dc(Classifier(unsupervised_x))])
    elif args.model_type == "Dcc_Da":
        Da = ModelDz().init(inputs=Input(shape=(args.dim_z,)))
        Encoder = ModelEncoderBN(layers_conv=args.conv_filters,
                                 is_bn=args.is_bn,
                                 dim_a=args.dim_a).init(inputs=Input(shape=(28, 28, 1)))
        Decoder = ModelDecoderBN(num_classes=args.classes).init(inputs=Input(shape=(args.dim_z,)))

        model = Model(unsupervised_x, [Decoder(Encoder(unsupervised_x)),
                                       Da(Encoder(unsupervised_x))])

    elif args.model_type == "Dcc_Da_Dc":
        Dc = ModelDc().init(inputs=input_c)
        Da = ModelDz().init(inputs=Input(shape=(args.dim_z,)))
        Encoder = ModelEncoderBN(layers_conv=args.conv_filters,
                                 is_bn=args.is_bn,
                                 dim_a=args.dim_a).init(inputs=Input(shape=(28, 28, 1)))
        Decoder = ModelDecoderBN(num_classes=args.classes).init(inputs=Input(shape=(args.dim_z,)))

        model = Model(unsupervised_x, [Decoder(Encoder(unsupervised_x)),
                                       Da(Encoder(unsupervised_x)),
                                       Dc(Decoder(Encoder(unsupervised_x)))])

    elif args.model_type in ["Dcc_Dz_Dc_Dxx", "Dcc_Dz_Dc_Dxx_Dx"]:
        Encoder = ModelEncoderZC(layers_conv=args.conv_filters,
                                 layers_dense=args.dense_filters,
                                 is_bn=args.is_bn,
                                 dim_z=args.dim_z,
                                 num_classes=args.classes).init(inputs=Input(shape=(28, 28, 1)))

        input_z, input_c = Encoder(unsupervised_x)
        model = Model(unsupervised_x, [input_c, input_z])

    # === Test =========================================================================================================
    Epochs = []
    Errors = []
    for epoch in range(1, args.epochs+1):
        if epoch >= 1000 and epoch % 500 != 0:
            continue
        elif epoch >= 100 and epoch % 100 != 0:
            continue
        elif epoch >= 10 and epoch % 10 != 0:
            continue

        # load trained model weights
        model.load_weights("%s/model_epoch_%d" % (checkpoint_dir_, epoch))

        error = 0
        n     = 0
        cur_epoch = test_data.epochs_completed
        while cur_epoch == test_data.epochs_completed:

            batch_data = test_data.next_batch_dict()
            input = batch_data['im']
            label = batch_data['label']

            n += input.shape[0]

            y_predict = model.predict(input)[0]
            predict = np.argmax(y_predict, axis=1)

            diff = abs(predict - label)
            diff[diff != 0] = 1
            error += np.sum(diff)

        Epochs.append(epoch)
        Errors.append(100*error/n)

        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t epoch : {epoch}, \t classification error = {100*error/n}")

    ind = np.argmin(np.asarray(Errors))
    print(f"\n\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t epoch : {Epochs[ind]}, \t classification error = {Errors[ind]}")

































