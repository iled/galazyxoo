# coding=utf-8
import os

import numpy as np
import pandas as pd
import skimage.io as sio
import skimage.transform as st
from sklearn.cluster import KMeans


def augment(img):
    transformed = cropping(img)
    resized = downsampling(transformed)
    noised = color_perturbation(resized)
    augmented = overlapping(noised)
    fixed = []
    for img in augmented:
        fixed.append(st.resize(img, (224, 224), mode='reflect'))
    return fixed


def cropping(img):
    # crop to 207x207 + rotate 45 deg + flips
    rotated = st.rotate(img, 45)[108:-108, 108:-108]
    cropped = img[108:-108, 108:-108]
    transformed = [cropped, np.fliplr(cropped), rotated, np.fliplr(rotated)]
    return transformed


def downsampling(transformed):
    # downsample to 69x69
    resized = [st.resize(cropped, (69, 69), mode='reflect') for cropped in
               transformed]
    return resized


def color_perturbation(resized):
    # color perturbation
    color_channel_weights = np.array([-0.0148366, -0.01253134, -0.01040762],
                                     dtype='float32')
    stds = np.random.randn(1).astype('float32') * 0.5  # std, chunk_size
    noise = stds * color_channel_weights  # [:, None]
    noised = [np.clip(img + noise, 0, 1) for img in resized]
    return noised


def overlapping(noised):
    # overlapping parts
    parts = []
    for img in noised:
        parts.append(img[:46, :46])
        parts.append(st.rotate(img[:46, 23:], 90))
        parts.append(st.rotate(img[23:, :46], -90))
        parts.append(st.rotate(img[23:, 23:], 180))

    return parts


def clusterize(data, save=None):
    kmeans = KMeans(3).fit(data.iloc[:, 1:4])
    ids = kmeans.labels_
    centroids = kmeans.cluster_centers_
    newset = pd.concat([data['GalaxyID'], pd.Series(ids, name='Class')],
                       axis=1)

    if save:
        newset.to_pickle(save)

    return newset, centroids


def set_class(data, nsamples=16, save=None):
    newset = pd.DataFrame(np.empty((data.shape[0] * nsamples),
                                   dtype=[('GalaxyID', str),
                                          ('Class', np.uint8)]))

    j = 0
    for i, row in data.iterrows():
        prob = row[1:4]
        x = np.random.rand(nsamples)
        cond = [x < prob[0], x < sum(prob[:2])]
        classes = [0, 1]
        cl = np.select(cond, classes, 2)
        gid = ['{0}_{1}'.format(int(row[0]), n) for n in xrange(nsamples)]
        try:
            newset.iloc[j:j + nsamples, 0] = gid
            newset.iloc[j:j + nsamples, 1] = cl
        except ValueError:
            print j, nsamples, len(gid)
            print newset.shape
            raise ValueError

        j += nsamples

    if save:
        newset.to_pickle(save)


def process_dir(basedir, newdir):
    if not os.path.exists(newdir):
        os.mkdir(newdir)

    for root, dirs, files in os.walk(basedir):
        if not dirs:
            l = len(files)
            for f, fn in enumerate(files):
                if fn.endswith('.jpg'):
                    if (f < l - 1 and os.path.exists(
                            os.path.join(newdir, files[f + 1].split('.')[0]))):
                        continue
                    img = sio.imread(os.path.join(basedir, fn))
                    imgs = augment(img)
                    fname = fn.split('.')[0]
                    for i, im in enumerate(imgs):
                        new_fn = os.path.join(newdir, fname,
                                              fname + '_{0}.jpg'.format(i))
                        if os.path.exists(new_fn):
                            continue
                        try:
                            os.mkdir(os.path.join(newdir, fname))
                        except OSError:
                            pass
                        sio.imsave(new_fn, im)


if __name__ == '__main__':
    basedir = '/afs/cs.pitt.edu/usr0/julio/private/kg/images_training_rev1'
    newdir = os.path.join(os.path.dirname(basedir), 'newgals')
    # process_dir(newdir)

    data = pd.read_csv(
        '/afs/cs.pitt.edu/usr0/julio/private/kg/training_solutions_rev1.csv')
    set_class(data, save='galaxies_3classes_sampling.npy')
