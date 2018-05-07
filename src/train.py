import pickle
import numpy as np
from PIL import Image
import os
from io import StringIO
import math
import pylab
from skimage import io
from skimage.transform import resize, rotate
from pathlib import Path

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L


import numpy


image_dir = './data/pokemon_images'
out_image_dir = './results/out_images'
out_model_dir = './results/out_models'


nz = 100          # # of dim for Z
batchsize=100
n_epoch=10000
n_train=200000
image_save_interval = 50000

# read all images
fs = os.listdir(image_dir)
print(len(fs))

dataset = []
for fn in fs:
    img = io.imread(Path(image_dir, fn))
    img = resize(img, (96, 96))
    dataset.append(img)
print(len(dataset))


class Generator(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l0z = L.Linear(nz, 6*6*512, initialW=0.02*math.sqrt(nz))
            self.dc1 = L.Deconvolution2D(
                512, 256, 4, stride=2, pad=1, initialW=0.02*math.sqrt(4*4*512))
            self.dc2 = L.Deconvolution2D(
                256, 128, 4, stride=2, pad=1, initialW=0.02*math.sqrt(4*4*256))
            self.dc3 = L.Deconvolution2D(
                128, 64, 4, stride=2, pad=1, initialW=0.02*math.sqrt(4*4*128))
            self.dc4 = L.Deconvolution2D(
                64, 3, 4, stride=2, pad=1, initialW=0.02*math.sqrt(4*4*64))
            self.bn0l = L.BatchNormalization(6*6*512)
            self.bn0 = L.BatchNormalization(512)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(64)

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(
            self.bn0l(self.l0z(z))), (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = (self.dc4(h))
        return x



class Discriminator(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(
                3, 64, 4, stride=2, pad=1, initialW=0.02*math.sqrt(4*4*3))
            self.c1 = L.Convolution2D(
                64, 128, 4, stride=2, pad=1, initialW=0.02*math.sqrt(4*4*64))
            self.c2 = L.Convolution2D(
                128, 256, 4, stride=2, pad=1, initialW=0.02*math.sqrt(4*4*128))
            self.c3 = L.Convolution2D(
                256, 512, 4, stride=2, pad=1, initialW=0.02*math.sqrt(4*4*256))
            self.l4l = L.Linear(6*6*512, 2, initialW=0.02*math.sqrt(6*6*512))
            self.bn0 = L.BatchNormalization(64)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)

    def __call__(self, x, test=False):
        h = F.elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = F.elu(self.bn1(self.c1(h)))
        h = F.elu(self.bn2(self.c2(h)))
        h = F.elu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return l


def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))


def train_dcgan_labeled(gen, dis, epoch0=0):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    zvis = (np.random.uniform(-1, 1, (100, nz)).astype(np.float32))

    for epoch in range(epoch0,n_epoch):
        perm = np.random.permutation(n_train)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)

        for i in range(0, n_train, batchsize):
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            x2 = np.zeros((batchsize, 3, 96, 96), dtype=np.float32)
            for j in range(batchsize):
                try:
                    rnd = np.random.randint(len(dataset))
                    rnd2 = np.random.randint(2)

                    img = np.asarray(dataset[rnd]) \
                            .astype(np.float32) \
                            .transpose(2, 0, 1)
                    if rnd2==0:
                        x2[j,:,:,:] = (img[:,:,::-1]-128.0)/128.0
                    else:
                        x2[j,:,:,:] = (img[:,:,:]-128.0)/128.0
                except Exception as e:
                    print('read image error occured', fs[rnd])
                    print(e)
            #print "load image done"

            # train generator
            z = Variable(np.random.uniform(-1, 1, (batchsize, nz)).astype(np.float32))
            x = gen(z)
            yl = dis(x)
            L_gen = F.softmax_cross_entropy(
                yl, Variable(np.zeros(batchsize, dtype=np.int32)))
            L_dis = F.softmax_cross_entropy(
                yl, Variable(np.ones(batchsize, dtype=np.int32)))

            # train discriminator

            x2 = Variable(x2)
            yl2 = dis(x2)
            L_dis += F.softmax_cross_entropy(
                yl2, Variable(np.zeros(batchsize, dtype=np.int32)))

            gen.cleargrads()
            L_gen.backward()
            o_gen.update()

            dis.cleargrads()
            L_dis.backward()
            o_dis.update()

            sum_l_gen += L_gen.data
            sum_l_dis += L_dis.data

            #print "backward done"

            if i%image_save_interval==0:
                pylab.rcParams['figure.figsize'] = (16.0,16.0)
                pylab.clf()
                vissize = 100
                z = zvis
                z[50:,:] = (np.random.uniform(-1, 1, (50, nz)).astype(np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x = x.data
                for i_ in range(100):
                    tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
                    pylab.subplot(10,10,i_+1)
                    pylab.imshow(tmp)
                    pylab.axis('off')
                pylab.savefig('%s/vis_%d_%d.png'%(out_image_dir, epoch,i))

        serializers.save_hdf5(
            "%s/dcgan_model_dis_%d.h5"%(out_model_dir, epoch),dis)
        serializers.save_hdf5(
            "%s/dcgan_model_gen_%d.h5"%(out_model_dir, epoch),gen)
        serializers.save_hdf5(
            "%s/dcgan_state_dis_%d.h5"%(out_model_dir, epoch),o_dis)
        serializers.save_hdf5(
            "%s/dcgan_state_gen_%d.h5"%(out_model_dir, epoch),o_gen)
        print('epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train)


gen = Generator()
dis = Discriminator()


try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
except:
    pass

train_dcgan_labeled(gen, dis)
