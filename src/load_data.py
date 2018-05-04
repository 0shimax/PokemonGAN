import random
import tensorflow as tf
from os import listdir
from os.path import splitext
from os.path import isfile, join
from pathlib import Path
import pandas as pd
from skimage.transform import resize, rotate
from skimage import io
from skimage.color import rgba2rgb
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
import cv2
from matplotlib import pyplot as plt
import numpy
numpy.random.seed(555)
random.seed(555)


def rotate_and_flip_images(imgs):
    rotated = numpy.empty_like(imgs, dtype=imgs.dtype)

    for i, img in enumerate(imgs):
        rotate_angle = random.choice([0., 3., 5., 7., 0., -3., -5., -7.])
        cols, rows, ch = img.shape
        rt_img = rotate(img, rotate_angle)
        # mask = (rt_img[:,:,0]<255) * (rt_img[:,:,1]<255) * (rt_img[:,:,2]<255)
        # dst1 = cv2.floodFill(rt_img, mask, (0, 0), (255, 255, 255))
        # dst1 = cv2.floodFill(dst1, mask, (cols, 0), (255, 255, 255))
        # dst1 = cv2.floodFill(dst1, mask, (0, rows), (255, 255, 255))
        # dst1 = cv2.floodFill(dst1, mask, (cols, rows), (255, 255, 255))
        if random.choice([True, False]):
            rotated[i] = rt_img[::-1, :, :]
        else:
            rotated[i] = rt_img
        # io.imshow(rt_img)
        # plt.show()
        # assert False, "test"
    return rotated


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
        self.i_end = None
        return self

    # def next_batch(self):
    def __next__(self):
        """Return the next `batch_size` examples from this data set."""
        if self.i_end is not None and self.i_end > self._num_examples:
            self.i_end = None
            raise StopIteration

        i = self.current_position
        self.i_end = i + self.batch_size

        # Shuffle wrong image index
        wrong_perm = numpy.random.permutation(self._num_examples)

        if self.i_end > self._num_examples:
            wrong_image_idx = wrong_perm[i:]
            real_images = self._images[i:]
            wrong_images = self._images[wrong_image_idx]
            captions = self._captions[i:]

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
            wrong_image_idx = wrong_perm[i:self.i_end]
            real_images = self._images[i:self.i_end]
            wrong_images = self._images[wrong_image_idx]
            captions = self._captions[i:self.i_end]
            self.current_position = self.i_end

        # real_images = rotate_and_flip_images(real_images)

        return (tf.convert_to_tensor(real_images),
               tf.convert_to_tensor(wrong_images),
               tf.convert_to_tensor(captions))


def get_images_and_captions(image_file_names, image_dir_path,
                            resized_image_size, df_data, caption_dim):

    def padding_ignore_tag(l_caption):
        l_caption = eval(l_caption)
        caption_len = len(l_caption)
        if caption_len < caption_dim:
            n_ignore_tag = caption_dim - caption_len
            return l_caption + ['<ignore>']*n_ignore_tag
        elif caption_len > caption_dim:
            return l_caption[:caption_dim]
        return l_caption

    # images = numpy.empty([len(image_file_names),
    #                       resized_image_size,
    #                       resized_image_size, 3], dtype=numpy.float32)
    images = []
    captions = []

    for idx, image_name in enumerate(image_file_names):
        try:
            img = io.imread(Path(image_dir_path, image_name))
            resized = resize(img, (resized_image_size, resized_image_size))

            pokemon_name, _ = splitext(image_name)
            caption = \
                df_data[df_data['name'] == pokemon_name].abilities.values[0]
            caption = padding_ignore_tag(caption)
            caption.insert(0, pokemon_name)
            captions.append(caption)
            images.append(resized)
        except:
            pass

    return numpy.array(images, dtype=numpy.float32), captions


def read_data_sets(data_root, image_file_dir, caption_file_name,
                   caption_dim, batch_size, resized_image_size,
                   wv_model):
    class DataSets(object):
        pass

    data_sets = DataSets()
    image_dir_path = Path(data_root, image_file_dir)

    image_file_names = [f for f in listdir(image_dir_path)
                        if isfile(join(image_dir_path, f)) and '-' not in f]

    caption_file_path = Path(data_root, caption_file_name)
    df_captions = pd.read_csv(caption_file_path)

    images, captions = get_images_and_captions(image_file_names,
                                               image_dir_path,
                                               resized_image_size,
                                               df_captions,
                                               caption_dim)

    n_data = len(images)
    captions = captions[:n_data]
    captions, voc_size = caption_to_one_hot(captions, wv_model)

    train_images, test_images, train_captions, test_captions = \
        train_test_split(images, captions,
                         random_state=55, shuffle=False, test_size=.15)
    data_sets.train = DataSet(train_images, train_captions, batch_size)
    data_sets.test = DataSet(test_images, test_captions, batch_size)

    return data_sets, voc_size


def caption_to_one_hot(captions, wv_model):
    vocab = Dictionary(captions)
    one_hot_matrix = numpy.array(
        [[wv_model.wv.vocab[word].index for word in words]
            for words in captions])
    return one_hot_matrix, len(vocab.keys())
