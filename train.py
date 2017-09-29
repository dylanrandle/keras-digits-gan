import os
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils.generic_utils import Progbar

from models import *

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--z_size', type=int, default=100, help='dimension of latent Z variable')
    parser.add_argument('--d_iters', type=int, default=5, help='num iterations to train D for every 1 iteration of G')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./save')

    args = parser.parse_args()

    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    if not os.path.isdir(args.log_dir): os.makedirs(args.log_dir)

    train(args)

# saves 10x10 sample of generated images
def generate_samples(args, step, n=0, save=True):
    zz = np.random.normal(0., 1., (100, args.z_size))
    generated_classes = np.array(list(range(0,10)) * 10)
    generated_images = G.predict([zz, generated_classes.reshape(-1, 1)])

    rr = []
    for c in range(10):
        rr.append(
            np.concatenate(generated_images[c * 10:(1 + c) * 10]).reshape(280, 28))
    img = np.hstack(rr)

    if save:
        plt.imsave(args.save_dir + '/samples_%07d.png' % n, img, cmap=plt.cm.gray)

    return img

def update_tb_summary(args, step, sw, D_true_losses, D_fake_losses, DG_losses, write_sample_images=True):
    s = tf.Summary()

    for names, vals in zip((('D_real_is_fake', 'D_real_class'),
                            ('D_fake_is_fake', 'D_fake_class'),
                            ('DG_is_fake', 'DG_class')),
                            (D_true_losses, D_fake_losses, DG_losses)):
        v = s.value.add()
        v.simple_value = vals[-1][1]
        v.tag = names[0]

        v = s.value.add()
        v.simple_value = vals[-1][2]
        v.tag = names[1]

    v = s.value.add()
    v.simple_value = -D_true_losses[-1][1] - D_fake_losses[-1][1]
    v.tag = 'D loss (-1*D_real_is_fake - D_fake_is_fake)'

    if write_sample_images:
        img = generate_samples(args, step, save=True)
        s.MergeFromString(tf.Session().run(
            tf.summary.image('samples_%07d' % step, img.reshape([1, *img.shape, 1]))))

    sw.add_summary(s, step)
    sw.flush()

def train(args):
    # load mnist data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # use all available 70k samples from both train and test sets
    X_train = np.concatenate((X_train, X_test))
    y_train = np.concatenate((y_train, y_test))
    # convert to -1..1 range, reshape to (sample_i, 28, 28, 1)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    sw = tf.summary.FileWriter(args.log_dir)
    progress_bar = Progbar(target=args.num_iterations)

    DG_losses = []
    D_true_losses = []
    D_fake_losses = []

    D = get_D()
    G = get_G(args)
    DG = get_DG(args)

    for i in range(args.num_iterations):

        if len(D_true_losses) > 0:
            progress_bar.update(
                i, 
                values=[
                    ('D_real_is_fake', np.mean(D_true_losses[-5:], axis=0)[1]),
                    ('D_real_class', np.mean(D_true_losses[-5:], axis=0)[2]),
                    ('D_fake_is_fake', np.mean(D_fake_losses[-5:], axis=0)[1]),
                    ('D_fake_class', np.mean(D_fake_losses[-5:], axis=0)[2]),
                    ('D(G)_is_fake', np.mean(DG_losses[-5:],axis=0)[1]),
                    ('D(G)_class', np.mean(DG_losses[-5:],axis=0)[2])
                ]
            )

        else: 
            progress_bar.update(i)

        # Step 1: train D on real+generated images

        if (i % 1000) < 25 or i % 500 == 0: # 25 times / 1000 or every 500th
            d_iters = 100
        else:
            d_iters = args.d_iters

        for d in range(d_iters):

            # unfreeze D
            D.trainable = True
            for l in D.layers: l.trainable = True

            # clip D weights
            for l in D.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -0.01, 0.01) for w in weights]
                l.set_weights(weights)

            # Step 1.1: maximize D output on reals <==> minimize -1*D(real)
            # randomly select from real images
            idx = np.random.choice(len(X_train), args.batch_size, replace=False)
            real_images = X_train[idx]
            real_images_classes = y_train[idx]

            D_loss = D.train_on_batch(real_images, [-np.ones(args.batch_size), real_images_classes])
            D_true_losses.append(D_loss)

            # Step 1.2: minimize D output on fakes
            zz = np.random.normal(0., 1., (args.batch_size, args.z_size))
            generated_classes = np.random.randint(0, 10, args.batch_size)
            generated_images = G.predict([zz, generated_classes.reshape(-1, 1)])

            D_loss = D.train_on_batch(generated_images, [np.ones(args.batch_size), generated_classes])
            D_fake_losses.append(D_loss)

        # Step 2: freeze D, and train D(G)
        # minimize D output while supplying it with fakes

        # freeze D 
        D.trainable = False
        for l in D.layers: l.trainable = False

        zz = np.random.normal(0., 1., (args.batch_size, args.z_size))
        generated_classes = np.random.randint(0, 10, args.batch_size)

        DG_loss = DG.train_on_batch(
            [zz, generated_classes.reshape((-1, 1))],
            [-np.ones(args.batch_size), generated_classes])
        DG_losses.append(DG_loss)

        if i % 10 == 0:
            update_tb_summary(args, i, sw, D_true_losses, D_fake_losses, DG_losses, write_sample_images=(i%250==0))

if __name__ == '__main__':
    main()