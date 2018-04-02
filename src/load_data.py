import gzip
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
import numpy
numpy.random.seed(555)


class DataSet(object):
    def __init__(self, images, captions):
        """Construct a DataSet.
           one_hot arg is used only if fake_data is true.
        """

        assert images.shape[0] == captions.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   captions.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)

        self._images = images
        self._captions = captions
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def captions(self):
        return self._captions

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, z_dim):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.random.permutation(self._num_examples)
            self._images = self._images[perm]
            self._captions = self._captions[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        diff = end - start
        z_noise = numpy.random.uniform(-1, 1, [batch_size, z_dim])[:diff]

        wrong_perm = numpy.random.permutation(self._num_examples)
        numpy.random.shuffle(wrong_perm)
        wrong_image_idx = wrong_perm[start:end]
        return self._images[start:end], self._images[wrong_image_idx], \
            self._captions[start:end],  z_noise


def read_data_sets(data_root, image_file_dir, caption_file_name, caption_dim):
    class DataSets(object):
        pass

    def padding_ignore_tag(l_caption):
        caption_len = len(l_caption)
        if caption_len < caption_dim:
            n_ignore_tag = caption_dim - caption_len
            return l_caption + ['<ignore>']*n_ignore_tag
        elif caption_len > caption_dim:
            return l_caption[:caption_dim]
        return l_caption

    data_sets = DataSets()
    image_dir_path = Path(data_root, image_file_dir)

    image_file_names = [f for f in listdir(image_dir_path)
                        if isfile(join(image_dir_path, f)) and '-' not in f]

    caption_file_path = Path(data_root, caption_file_name)
    df_captions = pd.read_csv(caption_file_path)
    captions = df_captions.abilities.apply(padding_ignore_tag).values
    print(captions)

    images = \
        [io.imread(Path(image_dir_path, image_name))
            for image_name in image_file_names]

    train_images, test_images, train_captions, test_captions = \
        train_test_split(images, captions,
                         random_state=55, shuffle=False, test_size=.15)
    data_sets.train = DataSet(train_images, train_captions)
    data_sets.test = DataSet(test_images, test_captions)

    return data_sets
