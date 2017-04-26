# coding=utf-8
from caferacer import CafeRacer
import os
import pickle

use_iter = 576
base = '/afs/cs.pitt.edu/usr0/julio/private'
data_sets = os.path.join(base, 'kg/data_sets.txt')
classes = os.path.join(base, 'kg/galaxies_3classes_sampling.npy')

deploy = os.path.join(base, 'kg_deployed/project/models/deploy.prototxt')

weights = os.path.join(base,
                       'kg_deployed/project/models/trained/_iter_{0}'
                       '.caffemodel'.format(use_iter))

espresso = CafeRacer(datasets_txt=data_sets, classes_npy=classes,
                     data_layer='loss3/classifier3',
                     device=0, arch=deploy, weights=weights)

espresso.run(train=True, test=True, val=True)

# save
espresso.save_state(prefix='svmer_{0}_'.format(use_iter))

# vs val set
espresso.run_svm(vs=espresso.val)
accuracy_val, confusion_matrix_val = espresso.svm_accuracy
print 'accuracy val: ', accuracy_val
print 'confusion val: ', confusion_matrix_val

# vs test set
espresso.run_svm(vs=espresso.test)
accuracy_test, confusion_matrix_test = espresso.svm_accuracy
print 'accuracy test: ', accuracy_test
print 'confusion test: ', confusion_matrix_test

