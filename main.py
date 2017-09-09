import os, time
import scipy.misc
import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from prepare_image import *
from model import *

image_size = 64
epochs = 20
learning_rate = 0.0002
beta = 0.5
iter_num = 2000
batch_size = 64
output_size = 64
sample_size = 64
c_dim = 3
sample_step = 500
save_step = 500
checkpoint_dir = 'checkpoint'
sample_dir = 'sample'
data_dir = 'actress'
is_train = False
visualize = False

data_files = load_image(data_dir)

def main():
    #pp.pprint(flags.FLAGS.__flags)

    tl.files.exists_or_mkdir(checkpoint_dir)
    tl.files.exists_or_mkdir(sample_dir)

    z_dim = 128
    with tf.device("/gpu:0"):
        ##========================= DEFINE MODEL ===========================##
        z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')
        real_images =  tf.placeholder(tf.float32, 
                                      [batch_size, output_size, output_size, c_dim], name='real_images')

        # z --> generator for training
        net_g, g_logits = generator_simplified_api(z, is_train=True, reuse=False)
        # generated fake images --> discriminator
        net_d, d_logits = discriminator_simplified_api(net_g.outputs, is_train=True, reuse=False)
        # real images --> discriminator
        net_d2, d2_logits = discriminator_simplified_api(real_images, is_train=True, reuse=True)
        # sample_z --> generator for evaluation, set is_train to False
        # so that BatchNormLayer behave differently
        net_g2, g2_logits = generator_simplified_api(z, is_train=False, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        # cost for updating discriminator and generator
        # discriminator: real images are labelled as 1
        d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
        # discriminator: images from generator (fake) are labelled as 0
        d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
        d_loss = d_loss_real + d_loss_fake
        # generator: try to make the the fake images look real (1)
        g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

        net_g.print_params(False)
        print("---------------")
        net_d.print_params(False)

        # optimizers for updating discriminator and generator
        d_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
        
        g_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
        '''
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1 = beta) \
                          .minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1 = beta) \
                          .minimize(g_loss, var_list=g_vars)
        '''

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    model_dir = "%s_%s_%s" % (data_dir, batch_size, output_size)
    save_dir = os.path.join(checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(sample_dir)
    tl.files.exists_or_mkdir(save_dir)
    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    #data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))

    sample_seed = np.random.normal(loc=0.0, scale=1.0, 
                                   size=(sample_size, z_dim)).astype(np.float32)
    # sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

    ##========================= TRAIN MODELS ================================##
    
    for epoch in range(epochs):
        iter_counter = 0
        sample_images = next_batch(sample_size, data_files)
        for idx in range(iter_num):
            #batch_z: random noise
            batch_z = np.random.normal(loc=0.0, scale=1.0, 
                                       size=(sample_size, z_dim)).astype(np.float32)  
            '''
            batch_z = np.random.uniform(low=-1, high=1, 
                                           size=(batch_size, z_dim)).astype(np.float32)
            '''
            #batch_images: real images
            batch_images = next_batch(batch_size, data_files)
            #print('batch_size:',np.shape(batch_images))
            start_time = time.time()
            # updates the discriminator
            errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images })
            # updates the generator, run generator twice to make sure that 
            #d_loss does not go to zero (difference from paper)
            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, epoch, idx, iter_num, time.time() - start_time, errD, errG))

            iter_counter += 1
            if np.mod(iter_counter, sample_step) == 0:
                # generate and visualize generated images
                img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], 
                                           feed_dict={z : sample_seed, real_images: sample_images})
                
                save_image_dir = os.path.join(sample_dir, 'epoch%d_iter%d'%(epoch, iter_counter))
                save_name = 'epoch%d_iter%d_'%(epoch, iter_counter)
                print_image(img, save_image_dir, save_name)
                '''
                tl.visualize.save_images(img, [8, 8], './{}/generated_image_{:02d}_{:04d}.png'\
                                         .format(sample_dir, epoch, idx))
                
                tl.visualize.save_images(batch_images, [8, 8], './{}/training_data_{:02d}_{:04d}.png'\
                                         .format(sample_dir, epoch, idx))
                '''
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))

            if np.mod(iter_counter, save_step) == 0:
                # save current network parameters
                print("[*] Saving checkpoints...")
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                print("[*] Saving checkpoints SUCCESS!")

if __name__ == '__main__':
    #tf.app.run()
    main()
