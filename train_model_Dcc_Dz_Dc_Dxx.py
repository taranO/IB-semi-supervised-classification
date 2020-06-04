'''
Variational Information Bottleneck for Semi-supervised Classification
with learnable latent space priors: Dcc + Dz + Dc + Dxx
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
from libs.ModelDz import ModelDz
from libs.ModelDc import ModelDc
from libs.ModelZC import ModelEncoderZC, ModelReconstructorZC

# ======================================================================================================================
print("PID = %d\n" % os.getpid())
print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================================================================================================================

parser = argparse.ArgumentParser(description="Variational Information Bottleneck for Semi-supervised Classification with learnable latent space priors: Dcc + Dz + Dc + Dxx")

parser.add_argument("--classes", default=10,   type=int, help="Number of classes")

parser.add_argument("--conv_filters",  default=[32,64,128], help="Encoder's convolutional layers")
parser.add_argument("--dense_filters", default=[1024], help="Encoder's dense layers")
parser.add_argument("--dim_z", default=10,   type=int, help="The dimensionality of latent space")
parser.add_argument("--is_bn", default=True, type=int, help="Is to apply batch normalisation in the encoder")

parser.add_argument("--lr",           default=1e-3, type=float, help="Training learning rate")
parser.add_argument("--weight_decay", default=0, type=float, help="Training weight decay")
parser.add_argument("--batch_size",   default=100,  type=int,   help="Training batch size") # !! must be even 10
parser.add_argument("--epochs",       default=100,   type=int,   help="Number of training epochs")

parser.add_argument("--checkpoint_dir", default="Dcc_Dz_Dc_Dxx/Dcc_Dz_Dc_Dxx_%sbn_%s_alpha_z_%s_alpha_c_%s_alpha_xx_%s_%snoise_%0.2f_run_%s", type=str, help="Dir where to same models")

parser.add_argument("--supervised_n", default=100, type=int, help="The amount of supervised data")
parser.add_argument("--alpha_z",      default=1, type=float, help="The regularization parameter for Dz")
parser.add_argument("--alpha_c",      default=1, type=float, help="The regularization parameter for Dc")
parser.add_argument("--alpha_xx",     default=1, type=float, help="The regularization parameter for Dxx")

parser.add_argument("--is_supervised_noise",  default=False, type=int, help="Is to use the stochastic encoder on supervised data")
parser.add_argument("--noise_std",            default=0.3, type=float, help="Noise std")
# !!! for the deterministic encoding --n_noise_realisations = 0
parser.add_argument("--n_noise_realisations", default=0,  type=int,   help="The number of of noise realisations")

parser.add_argument("--runs", default=3,   type=int, help="The number of runs")

args = parser.parse_args()

# ======================================================================================================================
if __name__ == "__main__":

    for run in range(1, args.runs+1):
        print("\n\n\n RUN %d \n\n\n" % run)
        dir = args.checkpoint_dir % ("" if args.is_bn else "no_",
                                     "all" if args.supervised_n == 0 else args.supervised_n,
                                     args.alpha_z,
                                     args.alpha_c,
                                     args.alpha_xx,
                                     "sup_" if args.is_supervised_noise else "",
                                     args.noise_std,
                                     run)

        checkpoint_dir_ = "checkpoints/%s" % dir
        makeDir(checkpoint_dir_)

        K.set_learning_phase(1)  # set learning phase: The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed as input to any Keras function that uses a different behavior at train time and test time.

        # --- print log --------------------------------------------------------
        info = {"checkpoint_dir_": checkpoint_dir_,
                '#_samples': "all" if args.supervised_n == 0 else args.supervised_n,
                '#_classes': args.classes, "dim_z": args.dim_z,
                "conv_filters": args.conv_filters, 'dense_filters': args.dense_filters,
                "is_bn": args.is_bn, "alpha_z": args.alpha_z, "alpha_c": args.alpha_c, "alpha_xx": args.alpha_xx,
                "is_supervised_noise": args.is_supervised_noise, "noise_std": args.noise_std, "n_noise_realisations": args.n_noise_realisations,
                'batch_size': args.batch_size, 'learning_rate': args.lr, 'weight_decay': args.weight_decay, 'epochs' : args.epochs}
        printToLog(info)

        # === load data ================================================================================================
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
        unsupervised_x = Input(shape=(28, 28, 1))
        supervised_x   = Input(shape=(28, 28, 1))
        input_z = Input(shape=(args.dim_z,))
        input_c = Input(shape=(args.classes,))

        Encoder = ModelEncoderZC(layers_conv=args.conv_filters,
                                 layers_dense=args.dense_filters,
                                 is_bn=args.is_bn,
                                 dim_z=args.dim_z,
                                 num_classes=args.classes).init(inputs=Input(shape=(28, 28, 1)))

        Reconstructor = ModelReconstructorZC().init(inputs=Input(shape=(args.classes + args.dim_z,)))

        # --------------------------------------------
        # Dz
        Dz = ModelDz().init(inputs=Input(shape=(args.dim_z,)))
        Dz_model = Model(input_z, Dz(input_z))
        Dz_model.compile(loss=keras.losses.binary_crossentropy,
                                 optimizer=keras.optimizers.Adam(learning_rate=args.lr)) # , decay=args.weight_decay
        Dz.trainable = False

        # Dc
        Dc = ModelDc().init(inputs=Input(shape=(args.classes,)))
        Dc_model = Model(input_c, Dc(input_c))
        Dc_model.compile(loss=keras.losses.binary_crossentropy,
                         optimizer=keras.optimizers.Adam(learning_rate=args.lr)) # , decay=args.weight_decay
        Dc.trainable = False

        #  mse
        input_z, input_c = Encoder(unsupervised_x)
        reconstructed_x = Reconstructor(keras.layers.concatenate([input_z, input_c]))
        MSE_model = Model(unsupervised_x, reconstructed_x)
        MSE_model.compile(loss=keras.losses.mse,
                      optimizer=keras.optimizers.Adam(learning_rate=args.lr)) # , decay=args.weight_decay

        # Dz + Dc
        unsupervised_z, unsupervised_c = Encoder(unsupervised_x)
        Dz_Dc_model = Model(unsupervised_x, [Dz(unsupervised_z), Dc(unsupervised_c)])
        Dz_Dc_model.compile(loss=[keras.losses.binary_crossentropy, keras.losses.binary_crossentropy],
                      optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                      loss_weights=[args.alpha_z, args.alpha_c])  # , decay=args.weight_decay

        # classifier
        Classifier_model = Model(supervised_x, [Encoder(supervised_x)[1], Encoder(supervised_x)[0]])
        Classifier_model.compile(loss=[keras.losses.categorical_crossentropy, keras.losses.binary_crossentropy],
                      optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                                                      loss_weights=[1, 0]) # , decay=args.weight_decay

        # === model scheme visualisation ===============================================================================
        results_dir_    = "results/%s" % dir
        makeDir(results_dir_)

        plot_model(Encoder, to_file="%s/model_Encoder_scheme.png" % results_dir_, show_shapes=True)
        plot_model(Reconstructor, to_file="%s/model_Reconstructor_scheme.png" % results_dir_, show_shapes=True)
        plot_model(Dz, to_file="%s/model_Dz_scheme.png" % results_dir_, show_shapes=True)
        plot_model(Dc, to_file="%s/model_Dc_scheme.png" % results_dir_, show_shapes=True)
        plot_model(Classifier_model, to_file="%s/model_classifier_scheme.png" % results_dir_, show_shapes=True)

        # === Training =================================================================================================
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
            Loss_Dz = []
            Loss_Dc = []
            Loss_mse = []

            cur_epoch = train_data_unlabel.epochs_completed
            while cur_epoch == train_data_unlabel.epochs_completed:

                batch_data = train_data_unlabel.next_batch_dict()
                x_unlabeled = batch_data['im']

                # --- MSE ---------------------------
                loss = MSE_model.train_on_batch(x_unlabeled, x_unlabeled)
                Loss_mse.append(loss)

                # Dz
                x = np.concatenate((np.random.normal(0, 1, size=[batch_size, args.dim_z]), Encoder.predict(x_unlabeled)[0]))
                # real images label is 1.0
                y = np.ones([2 * batch_size, 1])
                # fake images label is 0.0
                y[batch_size:, :] = 0.0
                loss_dz = Dz_model.train_on_batch(x, y)
                Loss_Dz.append(loss_dz)

                # Dy
                x = np.concatenate(
                    (np.tile(np.eye(10, dtype=int), (args.batch_size // 10, 1)), Encoder.predict(x_unlabeled)[1]))
                # real images label is 1.0
                y = np.ones([2 * batch_size, 1])
                # fake images label is 0.0
                y[batch_size:, :] = 0.0
                loss_dc = Dc_model.train_on_batch(x, y)
                Loss_Dc.append(loss_dc)

                # ---- Dz + Dc
                loss = Dz_Dc_model.train_on_batch(x_unlabeled, [np.ones([batch_size, 1]), np.ones([batch_size, 1])])

                # supervised part
                if args.supervised_n > 0:
                    batch_data = train_data_label.next_batch_dict()
                    x_labeled = batch_data['im']
                    labels = oneHotLabel(batch_data['label'])
                else:
                    x_labeled = x_unlabeled
                    labels = oneHotLabel(batch_data['label'])

                loss = Classifier_model.train_on_batch(x_labeled, [labels, np.random.normal(0, 1, size=[batch_size, args.dim_z])])
                Loss.append(loss)

            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t epoch : {epoch}, \t"
                  f"Dcc = {np.mean(np.asarray(Loss))}\t"
                  f"Dxx = {np.mean(np.asarray(Loss_mse))}\t"
                  f"Dz = {np.mean(np.asarray(Loss_Dz))}\t"
                  f"Dc = {np.mean(np.asarray(Loss_Dc))}")

            if epoch % save_each == 0 or epoch == args.epochs:
                # model.save("%s/model_bce_epoch_%d" % (checkpoint_dir_, epoch))
                Classifier_model.save_weights("%s/model_epoch_%d" % (checkpoint_dir_, epoch))

