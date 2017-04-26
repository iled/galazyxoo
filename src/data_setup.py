# coding=utf-8
"""
Júlio Caineta 2017

Prepare the data set

"""
import json
import os
from random import shuffle


def dispatch(sets, set_type, shuffled=True):
    for category, images in sets[set_type].iteritems():
        if shuffled:
            shuffle(images)
        for i, image in enumerate(images):
            if i > 3072:
                # debug: only send in the first N images
                raise StopIteration
            yield os.path.join(category, image)


def load(fpath):
    with open(fpath, 'r') as fp:
        sets = json.load(fp)

    return sets


def pick(basedir, p_validation=10, p_test=10):
    """

    :param basedir:
    :param p_validation:
    :param p_test:
    :return:
    """
    # hacked: removed category
    sets = {'training': {'all': []},
            'validation': {'all': []},
            'test': {'all': []}}

    for root, dirs, files in os.walk(basedir):
        # assuming that the directory with the images has no folders
        if not dirs:
            # hacked: removed category
            # category = os.path.basename(root)
            category = 'all'
            # remove hidden files and thumbs and then shuffle
            img_files = [fn for fn in files if
                         (fn.endswith('.jpg') or fn.endswith('.png'))]
            shuffle(img_files)
            n_validation = int(round(len(img_files) * (p_validation / 100.)))
            n_test = int(round(len(img_files) * (p_test / 100.)))
            n_training = len(img_files) - n_validation - n_test
            # make sure there is at least one (more?) training images
            if n_training < 1:
                raise IndexError('negative training length')
            # hacked: removed category
            sets['training'][category].extend(img_files[:n_training])
            sets['validation'][category].extend(img_files[
                                           n_training:n_training +
                                                      n_validation])
            sets['test'][category].extend(img_files[-n_test:])

    return sets


def save(sets, fpath, sort_keys=True, indent=4):
    with open(fpath, 'w') as fp:
        json.dump(sets, fp, sort_keys=sort_keys, indent=indent)


def save_imagedata(sets, outpath, basedir=None):
    """Save the parsed images according to the ImageData Layer specifications,
    which expect a .txt file with an image file path per line followed by the
    corresponding label, separated with a space.

    See: http://caffe.berkeleyvision.org/tutorial/layers/imagedata.html

    Parameters
    ----------
    sets: dict
        Dictionary with the parsed files and categories.
    outpath: string
        File path where the generated files will be saved.
    basedir: string, optional
        Base directory to include before the category folder.

    """
    basedir = basedir or ''
    for set_type in sets.iterkeys():
        filepaths = []
        for label, images in sets[set_type].iteritems():
            for image in images:
                filepaths.append(
                    os.path.join(basedir, label,
                                 image) + ' ' + label + os.linesep)

        with open(os.path.join(outpath, set_type + '.txt'), 'w') as fp:
            fp.writelines(filepaths)


def get_size(sets):
    sizes = {}
    for set_type in sets.iterkeys():
        sizes[set_type] = 0
        for category, images in sets[set_type].iteritems():
            sizes[set_type] += len(images)

    return sizes


if __name__ == '__main__':
    path = '/afs/cs.pitt.edu/usr0/julio/private/kg/'
    datasets = pick(os.path.join(path, 'images_training_rev1', 'all'))
    save(datasets, os.path.join(path, 'data_sets.txt'))
    # save_imagedata(datasets, path, path)
    # d2 = load(os.path.join(path, 'test_sets.txt'))
    print 'sizes: ', get_size(datasets)
    # for img in dispatch(d2, 'validation'):
    #     print os.path.join(path, img)
    #     raw_input()
    # print datasets == d2
