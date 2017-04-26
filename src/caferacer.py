# coding=utf-8
"""
Extract features, train network

Júlio Caineta 2017

"""

import numpy as np
import pandas as pd
import caffe
import os
import pickle
import sys
import time

base = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base)

import data_setup
from machinarium import Machinarium

os.chdir('/afs/cs.pitt.edu/usr0/julio/private/kg')


class CafeRacer(object):
    """
    Use a pretrained network as a feature extractor.

    """
    # set the default nomenclature for each type of data set
    test = 'test'
    train = 'training'
    val = 'validation'

    def __init__(self, device=0, arch=None, weights=None, solver_proto=None,
                 datasets_txt=None, classes_npy=None, basedir=None,
                 data_layer='fc8'):
        """
        Init in GPU mode.

        Parameters
        ----------
        device : integer, default 0
            The number of the GPU to use.
        arch : string, optional
            Path to the file containing the network structure (or
            architecture), which tells Caffe how the various network layers
            connect. If None, use the default deploy.prototxt.
        weights : string, optional
            Path to the file to load that contains th weights learned during
            training and copies those weights into the network structure
            created by the first argument.
        datasets_txt : string
            File name of the file containing the data sets JSON, created by
            data.setup.
        basedir : string
            Data directory.
        data_layer : string, default 'fc8'
            The name of the layer that data are extracted from.

        """
        caffe.set_device(device)
        caffe.set_mode_gpu()
        self.arch = arch or '/afs/cs.pitt.edu/usr0/julio/private/kg_deployed' \
                            '/project/models/deploy.prototxt'
        self.weights = weights or \
                       '/afs/cs.pitt.edu/usr0/julio/private/models' \
                       '/bvlc_googlenet/bvlc_googlenet.caffemodel'
        self.solver_proto = solver_proto or \
                            '/afs/cs.pitt.edu/usr0/julio/private/kg_deployed' \
                            '/project/models/quick_solver.prototxt'
        self.datasets_txt = datasets_txt or 'data_sets.txt'
        self.classes_npy = classes_npy or 'galaxies_3classes.npy'
        self.basedir = basedir or \
                       '/afs/cs.pitt.edu/usr0/julio/private/kg' \
                       '/images_training_rev1/'
        self.data_layer = data_layer
        # declare and init attributes
        self.output_features = {self.test: [], self.train: [], self.val: []}
        self.labels = {self.test: [], self.train: [], self.val: []}
        self.gal_class = None
        self.dims = {}
        self.training_set = None
        self.test_set = None
        self.validation_set = None
        self.datasets = None
        self.shuffle = None
        self.labels_enum = None
        self.mode = None
        self.batch_size = None
        self.net = None
        self.transformer = None
        self.i = None
        self.svm = None
        self.svm_accuracy = None
        self.solver = None
        self.solver_accuracy = None
        self.loss = None
        self.epoch_accuracy = None
        self.accurate = {self.test: [], self.train: [], self.val: []}
        self.net_accuracy = {self.test: [], self.train: [], self.val: []}
        self.iter = 0
        self.img_order = {self.test: [], self.train: [], self.val: []}

    def init_dispatchers(self):
        self.training_set = data_setup.dispatch(
            self.datasets, self.train, self.shuffle)
        self.test_set = data_setup.dispatch(
            self.datasets, self.test, self.shuffle)
        self.validation_set = data_setup.dispatch(
            self.datasets, self.val, self.shuffle)

    def load_datasets(self, filename, labels, shuffle=True):
        """

        Parameters
        ----------
        filename : string
        labels : string
        shuffle : bool, default True
            Shuffle the data sets when retrieving file names.

        """
        self.datasets = data_setup.load(filename)
        self.shuffle = shuffle
        self.init_dispatchers()
        # self.labels_enum = {label: i for i, label in
        #                     enumerate(set(self.datasets[self.train]))}
        # self.labels_enum = {}
        # self.gal_class = pd.read_pickle(labels)
        self.labels_enum = pd.read_pickle(labels)
        # for g in self.datasets[self.train]['all']:
        #     gid = g.split('.')[0]
        #     print gid
        #     gal_class = self.gal_class.loc[
        #         self.gal_class['GalaxyID'] == gid, 'Class'].values
        #     print gal_class
        #     self.labels_enum[g] = gal_class
        #     print self.labels_enum[g]
        #     raw_input()

    def load_network(self, arch=None, weights=None, mode=1, batch_size=32):
        """
        Load a pretrained CNN model.

        By default loads a model that has been trained on 1.4M images to
        classify images into 1000 classes.

        Parameters
        ----------
        arch : string, optional
            Path to the file containing the network structure (or
            architecture), which tells Caffe how the various network layers
            connect. If None, use the default deploy.prototxt.
        weights : string, optional
            Path to the file to load that contains th weights learned during
            training and copies those weights into the network structure
            created by the first argument.
        mode : integer, default 1
            Tells Caffe which mode is used to load the network. Default is
            in test mode (1), rather than train mode.
        batch_size : integer, default 8
            Reshape data blob in order to process images in batches.

        """
        self.arch = arch or self.arch
        self.weights = weights or self.weights
        self.mode = mode
        self.batch_size = batch_size
        self.net = caffe.Net(self.arch, self.weights, self.mode)
        self.reshape_net()

    def reshape_net(self, batch_size=None, target='net'):
        net = self.net if target == 'net' else self.solver.net
        batch_size = batch_size or self.batch_size
        self.dims[target] = net.blobs['data'].data.shape[1:]
        net.blobs['data'].reshape(batch_size, *self.dims[target])

    def setup_transformer(self, mean_path=None):
        """
        We need to preprocess each image before the CNN classifies it. Set
        up the Python data transformer.

        The transformer is only required when using a *deploy.prototxt*-like
        network definition, so without the Data Layer.

        """
        mean_path = mean_path or \
                    '/afs/cs.pitt.edu/usr0/julio/private/models' \
                    '/ilsvrc_2012_mean.npy'

        transf = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})
        # average over pixels to obtain the mean (BGR) pixel values
        # transf.set_mean('data', np.load(mean_path).mean(1).mean(1))
        # Caffe expects C x H x W, so transpose the data
        transf.set_transpose('data', (2, 0, 1))
        # swap channels from RGB to BGR
        transf.set_channel_swap('data', (2, 1, 0))
        # change range of pixel values from [0, 1] to [0, 255]
        transf.set_raw_scale('data', 255.0)
        self.transformer = transf

    def get_filename(self, set_type):
        """
        Get a file name from the given data set type.

        Parameters
        ----------
        set_type : string

        Returns
        -------
        string

        """
        if set_type == self.train:
            get_me = self.training_set
        elif set_type == self.test:
            get_me = self.test_set
        else:
            get_me = self.validation_set

        filename = next(get_me)
        self.labels[set_type].append(self.get_label(filename))
        return os.path.join(self.basedir, filename)

    def get_label(self, fn):
        gid = os.path.basename(fn).split('.')[0]
        return int(self.labels_enum.loc[
            self.labels_enum['GalaxyID'] == gid, 'Class'].values)

    def load_batch(self, set_type, onto):
        """
        Load batch of images to the specified net, either onto the
        network or onto the solver.

        The file names have to be preloaded.

        Each image is loaded as as float, preprocessed according to the
        specified transformer, then copied into the memory allocated for the
        net.

        Parameters
        ----------
        set_type : string
            Type of the set to be loaded: 'training', 'test', or 'validation'.
        onto : string
            Load the batch onto the 'net' or the 'solver'.

        """
        net = self.net if onto == 'net' else self.solver.net
        current_labels = []

        self.i = 0

        for i in xrange(self.batch_size):
            self.i = i
            filename = self.get_filename(set_type)
            # workaround to fix path
            basen, fn = os.path.split(filename)
            dn = fn.split('_')[0]
            filename = os.path.join(basen, dn, fn)
            img = caffe.io.load_image(filename)
            img = self.transformer.preprocess('data', img)
            net.blobs['data'].data[i, ...] = img
            current_labels.append(self.get_label(filename))
            # self.labels_enum[os.path.basename(filename)])
            self.img_order[set_type].append(fn)

        if onto == 'solver':
            net.blobs['label'].data[...] = current_labels

        self.scores(current_labels, set_type)

    def classify_batch(self, set_type, layer=None):
        """
        Perform the classification on the loaded images (run forward).
        The resulting features are saved.

        Parameters
        ----------
        set_type : string
        layer : string, optional
            The name of the layer that data are extracted from.

        """
        layer = layer or self.data_layer
        self.net.forward()
        data = self.net.blobs[layer].data.copy()
        self.output_features[set_type].append(data)

    def extract_features(self, set_type):
        """
        Extract all the features from a given data set.

        Parameters
        ----------
        set_type : string
            Type of the set to be loaded: 'training', 'test', or 'validation'.

        """
        self.reshape_net()
        while True:
            try:
                self.load_batch(set_type, 'net')
            except StopIteration:
                if self.i:
                    self.net.blobs['data'].reshape(self.i, *self.dims['net'])
                    self.net.reshape()
                    self.classify_batch(set_type)
                break
            else:
                self.classify_batch(set_type)

        # concatenate results
        stack = np.vstack(self.output_features[set_type])
        self.output_features[set_type] = stack
        self.net_accuracy[set_type] = np.mean(
            [np.mean(acc) for acc in self.accurate[set_type]])

    def run(self, train=True, test=True, val=False):
        """
        Extract features from the selected data sets. By default, will use
        both the training and the test sets.

        Parameters
        ----------
        train : bool, default True
            Extract features from the training set.
        test : bool, default True
            Extract features from the test set.

        Returns
        -------
        dict
            Dictionary containing one np.array for each data set that the
            features were extracted from.

        """
        self.setup_network()
        if train:
            print "running training set"
            self.extract_features(self.train)
        if test:
            print "running test set"
            self.extract_features(self.test)
        if val:
            print "running validation set"
            self.extract_features(self.val)

        return self.output_features

    def setup_network(self):
        """
        Set up network using default values.

        """
        self.load_datasets(self.datasets_txt, self.classes_npy)
        self.load_network()
        self.setup_transformer()

    def run_svm(self, vs=None):
        """
        Run the SVM classifier. The Linear SVC is trained in the features
        extracted from the training set, and the it is tested with the
        features extracted from the test set. The accuracy of the prediction
        is computed (accuracy score and confusion matrix).

        Parameters
        ----------
        vs : string
            Run the SVM classifier against which data set. Default to the test
            set.

        Returns
        -------
        float, array, shape = [n_classes, n_classes]
            Fraction of correctly classified samples, and confusion matrix.

        """
        vs = vs or self.test

        self.svm = Machinarium(self.output_features[self.train],
                               self.labels[self.train],
                               self.output_features[vs],
                               self.labels[vs])

        self.svm_accuracy = self.svm.run()

        return self.svm_accuracy

    def save_state(self, prefix='cafe_'):
        """
        Dump output features and labels to pickles.

        Parameters
        ----------
        prefix : string

        """
        pickle.dump(self.output_features,
                    open(prefix + 'output_features.pckl', 'w'))
        pickle.dump(self.labels, open(prefix + 'labels.pckl', 'w'))
        pickle.dump(self.img_order, open(prefix + 'img_order.pckl', 'w'))

    def setup_solver(self, proto=None):
        self.setup_network()
        self.solver_proto = proto or self.solver_proto
        self.solver = caffe.SGDSolver(self.solver_proto)
        # we don't have enough data to train the network entirely from
        # scratch, so we will initialize the network to the same weights we
        # used before
        self.solver.net.copy_from(self.weights)
        # accuracy in each epoch
        self.solver_accuracy = []

    def run_solver(self, epochs=25, batch_size=8, outfile='brewed.caffemodel'):
        self.batch_size = batch_size
        self.loss = []

        for epoch in xrange(epochs):
            self.iter = 0
            start_time = time.time()
            print 'epoch: ', epoch
            self.init_dispatchers()
            print '\tloading batch to solver on epoch ', epoch
            self.solver_batch()
            elapsed_time = time.time() - start_time
            print '\tfinished solver batch. number of iterations: ', self.iter
            print '\tsolver elapsed time: ', elapsed_time
            self.iter = 0
            start_time = time.time()
            print 'run validation on epoch ', epoch
            self.run_validation()
            elapsed_time = time.time() - start_time
            print '\tfinished validation batch. number of iterations: ', \
                self.iter
            print '\tvalidation elapsed time: ', elapsed_time
            # keep alive
            try:
                os.system('krenew;aklog')
            except:
                pass

        self.solver.net.save(outfile)
        pickle.dump((self.solver_accuracy, self.loss),
                    open('acc_and_loss.pickle', 'w'))

    def solver_mini_batch(self):
        # train network and update of the weights using the minibatch
        self.solver.step(1)
        # get value of the loss
        self.loss.append(
            self.solver.net.blobs['loss3/classifier3'].data.copy())

    def solver_batch(self):
        self.reshape_net(target='solver')

        try:
            while True:
                self.iter += 1
                self.load_batch(self.train, 'solver')
                self.solver_mini_batch()
        except StopIteration:
            if self.i:
                print '\tfinished iterations'
                # self.solver.net.blobs['data'].reshape(self.i,
                #                                       *self.dims['solver'])
                # self.solver.net.reshape()
                self.solver_mini_batch()

        print '\tdone'

    def validation_batch(self):
        """
        Similar to classify_batch.

        """
        self.solver.net.forward()
        # save the accuracy, not the data -- we don't need it
        accuracy = self.solver.net.blobs['loss3/top-1'].data.copy()
        self.epoch_accuracy.extend(accuracy.flatten())

    def run_validation(self):
        """
        Run validation set in the solver.

        """
        self.reshape_net(target='solver')
        # accuracy in this specific epoch
        self.epoch_accuracy = []
        try:
            while True:
                self.iter += 1
                self.load_batch(self.val, 'solver')
                self.validation_batch()
        except StopIteration:
            if self.i:
                # self.solver.net.blobs['data'].reshape(self.i,
                #                                       *self.dims['solver'])
                # self.solver.net.reshape()
                self.validation_batch()

        self.solver_accuracy.append(np.mean(self.epoch_accuracy))

    def scores(self, true_labels, set_type):
        """
        Compare the network classification for the last images with their
        true labels. The result is accumulated in a list which is later
        averaged.

        Parameters
        ----------
        true_labels : array-like
            True labels of the last images classified in the network.
        set_type : string

        """
        data = self.net.blobs[self.data_layer].data
        predicted = np.array([img.argmax() for img in data])
        self.accurate[set_type].append(predicted == true_labels)
