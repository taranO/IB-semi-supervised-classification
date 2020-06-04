import numpy as np
import os

from libs.mnist import MNISTData

# ======================================================================================================================
def printToLog(info):
    for k, v in info.items():
        print(k + ' = ' + str(v))

    print("\n")

def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def oneHotLabel(labels):

    onehot_labels = []
    for value in labels:
        onehot_label = [0 for _ in range(10)]
        onehot_label[value] = 1
        onehot_labels.append(onehot_label)

    return np.asarray(onehot_labels).astype(np.float32)

def loadTrainData(batch_size, data_path_, n_use_label=None, n_use_sample=None):
    """ Function for load training data
    If n_use_label or n_use_sample is not None, samples will be
    randomly picked to have a balanced number of examples
    Args:
        batch_size (int): batch size
        n_use_label (int): how many labels are used for training
        n_use_sample (int): how many samples are used for training
    Retuns:
        MNISTData
    """
    data = MNISTData('train',
                     data_dir=data_path_,
                     shuffle=True,
                     n_use_label=n_use_label,
                     n_use_sample=n_use_sample,
                     batch_dict_name=['im', 'label'])
    data.setup(epoch_val=0, batch_size=batch_size)

    return data

def loadTrainDataWithNoise(batch_size, data_path_, n_use_label=None, n_use_sample=None, noise_std=0.3, n_noise_realisations = 3):
    """ Function for load training data
    If n_use_label or n_use_sample is not None, samples will be
    randomly picked to have a balanced number of examples
    Args:
        batch_size (int): batch size
        n_use_label (int): how many labels are used for training
        n_use_sample (int): how many samples are used for training
    Retuns:
        MNISTData
    """
    data = MNISTData('train',
                     data_dir=data_path_,
                     shuffle=True,
                     n_use_label=n_use_label,
                     n_use_sample=n_use_sample,
                     batch_dict_name=['im', 'label'])
    data.setup(epoch_val=0, batch_size=batch_size)

    n_use_sample, h, w, c = data.im_list.shape

    Img_list  = []
    Img_label = []
    Img_list.append(data.im_list)
    Img_label.append(data.label_list)
    for i in range(n_noise_realisations):
        noise = np.random.normal(0, noise_std, (n_use_sample, h, w, c))
        noisy_img = data.im_list+noise
        noisy_img[noisy_img < 0] = 0
        noisy_img[noisy_img > 1] = 1

        Img_list.append(noisy_img)
        Img_label.append(data.label_list)


    Images  = np.reshape(np.asarray(Img_list), (-1, h, w, c))
    Labeles = np.reshape(np.asarray(Img_label), (-1))

    shuffle_indx = np.arange(Images.shape[0])
    np.random.shuffle(shuffle_indx)

    data.im_list    = Images[shuffle_indx]
    data.label_list = Labeles[shuffle_indx]

    return data


def loadTestData(batch_size, data_path_):
    """ Function for load validation data """
    data = MNISTData('test',
                     data_dir=data_path_,
                     shuffle=True,
                     batch_dict_name=['im', 'label'])
    data.setup(epoch_val=0, batch_size=batch_size)
    return data