# coding=utf-8
"""
SVM classifier

Júlio Caineta 2017

"""
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix, accuracy_score


class Machinarium(object):
    """
    Train and test a linear SVM.

    """

    def __init__(self, x, x_labels, y, y_labels):
        self.x = x
        self.x_labels = x_labels
        self.y = y
        self.y_labels = y_labels
        self.classifier = svm.LinearSVC()
        # init
        self.scaler = None
        self.x_scaled = None
        self.y_scaled = None
        self.y_classified = None
        self.accuracy_score = None
        self.confusion_matrix = None

    def scale_x(self):
        """
        Standardize the training set with zero mean and unit variance.

        """
        self.scaler = preprocessing.StandardScaler().fit(self.x)
        self.x_scaled = self.scaler.transform(self.x)

    def scale_y(self):
        """
        Standardize the test set with zero mean and unit variance (of the
        training set).

        """
        self.y_scaled = self.scaler.transform(self.y)

    def scale(self):
        """
        Standardize the training and the test sets with zero mean and unit
        variance (of the training set).

        """
        self.scale_x()
        self.scale_y()

    def fit(self):
        """
        Fit the model according to the given training data.
        Use the scaled data set.

        """
        self.classifier.fit(self.x_scaled, self.x_labels)

    def predict(self):
        """
        Predict the class labels for the samples in the test set.
        Use the scaled data set.

        """
        self.y_classified = self.classifier.predict(self.y_scaled)

    def accuracy(self):
        """
        Compute the accuracy score and the confusion matrix to evaluate the
        accuracy of a classification.

        """

        self.accuracy_score = accuracy_score(self.y_labels,
                                             self.y_classified)
        self.confusion_matrix = confusion_matrix(self.y_labels,
                                                 self.y_classified)

    def run(self):
        """
        Train the classifier, test it, and compute the prediction accuracy.

        Returns
        -------
        float, array, shape = [n_classes, n_classes]
            Fraction of correctly classified samples, and confusion matrix.

        """
        self.scale()
        self.fit()
        self.predict()
        self.accuracy()

        return self.accuracy_score, self.confusion_matrix
