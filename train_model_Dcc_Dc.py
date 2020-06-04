'''
Variational Information Bottleneck for Semi-supervised Classification with no priors on latent space: Dcc + Dc
'''
from __future__ import print_function
import keras
from keras.layers import Input
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model

import argparse
import datetime

from libs.utils import *

from libs.ModelDcc import  ModelDcc
from libs.ModelDc import ModelDc

# ======================================================================================================================
print("PID = %d\n" % os.getpid())
print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ======================================================================================================================

parser = argparse.ArgumentParser(description="Variational Information Bottleneck for Semi-supervised Classification with no priors on latent space: Dcc + Dc")

parser.add_argument("--classes", default=10,   type=int, help="Number of classes")

parser.add_argument("--conv_filters",  default=[32,64,128], help="Encoder's convolutional layers")
parser.add_argument("--dense_filters", default=[1024,500], help="Encoder's dense layers")
parser.add_argument("--is_bn",         default=False, type=int, help="Is to apply batch normalisation in the encoder")

parser.add_argument("--lr",           default=1e-3, type=float, help="Training learning rate")
parser.add_argument("--weight_decay", default=1e-3, type=float, help="Training weight decay")
parser.add_argument("--batch_size",   default=100,  type=int,   help="Training batch size") # !! must be even 10
parser.add_argument("--epochs",       default=100,  type=int,   help="Number of training epochs")

parser.add_argument("--supervised_n",   default=100,    type=int, help="The amount of the supervised data")
parser.add_argument("--alpha_c",        default=0.0005, type=float, help="The regularization parameter for Dc")
parser.add_argument("--checkpoint_dir", default="Dcc_Dc/Dcc_Dc_%sbn_%s_%s_%snoise_%0.2f_run_%s",   type=str, help="Dir where to store the models")

parser.add_argument("--is_supervised_noise",  default=False, type=int, help="Is to use the stochastic encoder on supervised data")
parser.add_argument("--noise_std",            default=0.3, type=float, help="Noise std")
# !!! for the deterministic encoding --n_noise_realisations = 0
parser.add_argument("--n_noise_realisations", default=0,  type=int,   help="The number of of noise realisations")

parser.add_argument("--runs", default=3,   type=int, help="The number of runs")

args = parser.parse_args()

# ======================================================================================================================
if __name__ == "__main__":

    for run in range(1, args.runs + 1):
        print("\n\n\n RUN %d \n\n\n" % run)

        checkpoint_dir = args.checkpoint_dir % ("" if args.is_bn else "no_",
                                                "all" if args.supervised_n == 0 else args.supervised_n,
                                                args.alpha_c,
                                                "sup_" if args.is_supervised_noise else "",
                                                args.noise_std,
                                                run)
        checkpoint_dir_ = "checkpoints/%s" % checkpoint_dir
        makeDir(checkpoint_dir_)

        K.set_learning_phase(1)  # set learning phase: The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed as input to any Keras function that uses a different behavior at train time and test time.

        # --- print log --------------------------------------------------------
        info = {"checkpoint_dir_": checkpoint_dir_,
                '#_samples': "all" if args.supervised_n == 0 else args.supervised_n,
                '#_classes': args.classes, "conv_filters": args.conv_filters, 'dense_filters': args.dense_filters, "is_bn": args.is_bn, "alpha_c": args.alpha_c,
                "is_supervised_noise": args.is_supervised_noise, "noise_std": args.noise_std, "n_noise_realisations": args.n_noise_realisations,
                'batch_size': args.batch_size, 'learning_rate': args.lr, 'weight_decay': args.weight_decay, 'epochs' : args.epochs}
        printToLog(info)

        # ==========================================================

        # load data
        if args.supervised_n > 0:
            batch_size = args.batch_size if args.supervised_n >= args.batch_size else args.supervised_n//2
        else:
            batch_size = args.batch_size

        if not args.is_supervised_noise and args.n_noise_realisations > 0:
            train_data_unlabel = loadTrainDataWithNoise(batch_size, data_path_="./data", noise_std=args.noise_std, n_noise_realisations=args.n_noise_realisations)
        else:
            train_data_unlabel = loadTrainData(batch_size, data_path_="./data")

        if args.supervised_n > 0:
            if args.is_supervised_noise and args.n_noise_realisations > 0:
                train_data_label = loadTrainDataWithNoise(batch_size, data_path_="./data", noise_std=args.noise_std,
                                                           n_noise_realisations=args.n_noise_realisations, n_use_sample=args.supervised_n)
            else:
                train_data_label = loadTrainData(batch_size, data_path_="./data", n_use_sample=args.supervised_n)

        # ------------------------------------------------------------------------
        # init model
        unsupervised_x = Input(shape=(28, 28, 1))
        supervised_x   = Input(shape=(28, 28, 1))
        input_c        = Input(shape=(args.classes,))

        Dc = ModelDc().init(inputs=input_c)
        Classifier = ModelDcc(num_classes=args.classes,
                              layers_conv=args.conv_filters,
                              layers_dense=args.dense_filters,
                              is_bn=args.is_bn).init(inputs=Input(shape=(28, 28, 1)))

        # ------------------------------------------------------------------
        # 1.
        Dc_model = Model(input_c, Dc(input_c))
        Dc_model.compile(loss=keras.losses.binary_crossentropy,
                                 optimizer=keras.optimizers.Adam(learning_rate=args.lr, decay=args.weight_decay),
                                 metrics=['accuracy'])
        Dc.trainable = False

        # 2.
        Classifier_model = Model([supervised_x, unsupervised_x], [Classifier(supervised_x), Dc(Classifier(unsupervised_x))])

        Classifier_model.compile(loss=[keras.losses.categorical_crossentropy, keras.losses.binary_crossentropy],
                      optimizer=keras.optimizers.Adam(learning_rate=args.lr, decay=args.weight_decay),
                      metrics=['accuracy'], loss_weights=[1, args.alpha_c])

        # ==============================================================================
        # model scheme visualisation
        results_dir_ = "results/%s" % checkpoint_dir
        makeDir(results_dir_)

        plot_model(Classifier, to_file="%s/model_scheme.png" % results_dir_, show_shapes=True)
        plot_model(Dc, to_file="%s/model_dc_scheme.png" % results_dir_, show_shapes=True)
        plot_model(Classifier_model, to_file="%s/model_classifier_scheme.png" % results_dir_, show_shapes=True)

           # === Training ======================
        for epoch in range(1, args.epochs+1):

            if epoch <= 10:
                save_each = 1
            elif epoch <= 100:
                save_each = 10
            elif epoch <= 1000:
                save_each = 100
            elif epoch <= 10000:
                save_each = 500

            Loss = []
            Loss_dc = []

            i = 0
            cur_epoch = train_data_unlabel.epochs_completed
            while cur_epoch == train_data_unlabel.epochs_completed:

                batch_data = train_data_unlabel.next_batch_dict()
                x_unlabeled = batch_data['im']

                # 1.
                x = np.concatenate((np.tile(np.eye(10, dtype=int), (args.batch_size//10, 1)), Classifier.predict(x_unlabeled)))
                # real images label is 1.0
                y = np.ones([2 * batch_size, 1])
                # fake images label is 0.0
                y[batch_size:, :] = 0.0
                loss_dc = Dc_model.train_on_batch(x, y)
                Loss_dc.append(loss_dc)

                # 2.
                if args.supervised_n > 0:
                    batch_data = train_data_label.next_batch_dict()
                    x_labeled = batch_data['im']
                    labels = oneHotLabel(batch_data['label'])
                else:
                    x_labeled = x_unlabeled
                    labels = oneHotLabel(batch_data['label'])

                loss = Classifier_model.train_on_batch([x_labeled, x_unlabeled], [labels, np.ones([batch_size, 1])])
                Loss.append(loss)

            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t epoch : {epoch}, \t"
                  f"Dcc = {np.mean(np.asarray(Loss))}\t"
                  f"Dc = {np.mean(np.asarray(Loss_dc))}")

            if epoch % save_each == 0 or epoch == args.epochs:
                # model.save("%s/model_bce_epoch_%d" % (checkpoint_dir_, epoch))
                Classifier_model.save_weights("%s/model_epoch_%d" % (checkpoint_dir_, epoch))

