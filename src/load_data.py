import tensorflow as tf
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pandas as pd
from skimage.transform import resize
from skimage import io
from skimage.color import rgba2rgb
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
import numpy
numpy.random.seed(555)


class DataSet(tf.data.Dataset):
    def __init__(self, images, captions, batch_size):
        """Construct a DataSet.
           one_hot arg is used only if fake_data is true.
        """
        super().__init__()

        assert images.shape[0] == captions.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   captions.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 3
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)

        self._images = images
        self._captions = captions
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.batch_size = batch_size
        self.current_position = 0

    @property
    def images(self):
        return self._images

    @property
    def captions(self):
        return self._captions

    @property
    def num_examples(self):
        return self._num_examples

    def __iter__(self):
        """Returns self."""
        return self

    # def next_batch(self):
    def __next__(self):
        """Return the next `batch_size` examples from this data set."""
        i = self.current_position
        i_end = i + self.batch_size

        # Shuffle wrong image index
        wrong_perm = numpy.random.permutation(self._num_examples)

        if i_end > self._num_examples:
            wrong_image_idx = wrong_perm[i:]
            real_images = tf.convert_to_tensor(self._images[i:])
            wrong_images = tf.convert_to_tensor(self._images[wrong_image_idx])
            captions = tf.convert_to_tensor(self._captions[si:])

            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.random.permutation(self._num_examples)
            self._images = self._images[perm]
            self._captions = self._captions[perm]
            # Start next epoch
            assert self.batch_size <= self._num_examples
            self.current_position = 0
        else:
            real_images = tf.convert_to_tensor(self._images[i:i_end])
            wrong_images = tf.convert_to_tensor(self._images[i:i_end])
            captions = tf.convert_to_tensor(self._captions[i:i_end])
            self.current_position = i_end
        return real_images, wrong_images, captions


def read_data_sets(data_root, image_file_dir, caption_file_name,
                   caption_dim, batch_size, resized_image_size):
    class DataSets(object):
        pass

    def padding_ignore_tag(l_caption):
        l_caption = eval(l_caption)
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

    images = [resize(rgba2rgb(io.imread(Path(image_dir_path, image_name))),
                  (resized_image_size, resized_image_size))
              for image_name in image_file_names]
    images = numpy.array(images, dtype=numpy.float32)

    n_data = len(images)
    captions = captions[:n_data]
    captions, voc_size = caption_to_one_hot(captions)

    train_images, test_images, train_captions, test_captions = \
        train_test_split(images, captions,
                         random_state=55, shuffle=False, test_size=.15)
    data_sets.train = DataSet(train_images, train_captions, batch_size)
    data_sets.test = DataSet(test_images, test_captions, batch_size)

    return data_sets, voc_size


def caption_to_one_hot(captions):
    vocab = Dictionary(captions)
    one_hot_matrix = numpy.array([[vocab.token2id[word] \
        for word in words] for words in captions])
    return one_hot_matrix, len(vocab.keys())

# def get_training_batch(data_set, image_size, z_dim):
#     real_images, wrong_images, captions = data_set.next_batch()
# 	z_noise = np.random.uniform(-1, 1, [data_set.batch_size, z_dim])
# 	return real_images, wrong_images, captions, z_noise
