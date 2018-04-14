import gensim
from gensim.models import word2vec
from os import listdir
from os.path import splitext
from os.path import isfile, join
from pathlib import Path
import numpy as np
import pandas as pd


def read_data(data_root, image_file_dir, caption_file_name):
    """Extract the first file enclosed in a zip file as a list of words."""
    image_dir_path = Path(data_root, image_file_dir)

    image_file_names = [f for f in listdir(image_dir_path)
                        if isfile(join(image_dir_path, f)) and '-' not in f]

    caption_file_path = Path(data_root, caption_file_name)
    df_captions = pd.read_csv(caption_file_path)

    captions = create_caption_list(image_file_names, df_captions)
    return captions


def create_caption_list(image_file_names, df_data):
    captions = []

    for idx, image_name in enumerate(image_file_names):
        try:
            pokemon_name, _ = splitext(image_name)
            caption = \
                df_data[df_data['name'] == pokemon_name].abilities.values[0]
            caption = eval(caption)
            caption.insert(0, pokemon_name)
            caption += ['<ignore>']*2
            captions.append(caption)
        except:
            pass
    return captions


def create_embedding_matrix(args):
    """
    ref: http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
    """
    sentences = read_data(args['data_root'], args['image_file_dir'],
                          args['caption_file_name'])
    model = word2vec.Word2Vec(
        sentences, iter=500, window=3, min_count=1,
        size=args['vec_dim'], workers=4, seed=555)

    # save and reload the model
    model.save(args['output_dir'] + "/wv.model")
    # model = gensim.models.Word2Vec.load(root_path + "mymodel")

    embedding_matrix = np.zeros((len(model.wv.vocab), args['vec_dim']))
    for i in range(len(model.wv.vocab)):
        word = model.wv.index2word[i]
        # print(model.wv.vocab[word].index)
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def main(args):
    embedding_matrix = create_embedding_matrix(args)
    np.savez(Path(args['output_dir'], 'embedding_matrix'),
             weight=embedding_matrix)


if __name__ == '__main__':
    args = {
        'data_root': './data',
        'image_file_dir': 'pokemon_images',
        'caption_file_name': 'pokemon.csv',
        'output_dir': './data/embedding',
        'vec_dim': 128}

    main(args)
