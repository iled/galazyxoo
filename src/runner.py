# coding=utf-8
from caferacer import CafeRacer

data_sets = '/afs/cs.pitt.edu/usr0/julio/private/kg/data_sets.txt'
classes = '/afs/cs.pitt.edu/usr0/julio/private/kg/galaxies_3classes_sampling.npy'

# cafe = CafeRacer(datasets_txt=data_sets, classes_npy=classes)
# cafe.run(train=True, test=False)

solver = CafeRacer(datasets_txt=data_sets, classes_npy=classes,
                   data_layer='loss3/classifier3',
                   device=3)
solver.setup_solver()
solver.run_solver(epochs=30,
                  batch_size=128,
                  outfile='brewed2.caffemodel')