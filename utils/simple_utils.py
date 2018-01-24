import matplotlib.pyplot as plt
import numpy as np


def plot_batch_train(model, img_size, batch_size, sketch, color, epoch, idx, tag, nb_img=5):
    img_sketch = np.array(sketch[0:nb_img])
    img_color = np.array(color[0:nb_img])
    img_gen = model.predict(sketch, batch_size=batch_size)[0][0:nb_img]
    for i in range(nb_img):
        plt.subplot(nb_img, 3, i * 3 + 1)
        plt.imshow(img_sketch[i].reshape((img_size,img_size)), cmap='Greys_r')
        plt.axis('off')
        plt.subplot(nb_img, 3, i * 3 + 2)
        plt.imshow(img_color[i])
        plt.axis('off')
        plt.subplot(nb_img, 3, i * 3 + 3)
        plt.imshow(img_gen[i])
        plt.axis('off')
    plt.savefig("figures/%s_fig_epoch%s_idx%s.png" % (tag, epoch, idx))
    plt.clf()
    plt.close()


def plot_batch_eval(model, img_size, batch_size, sketch, tag, nb_img=16):
    img_sketch = np.array(sketch[0:nb_img])
    img_gen = model.predict(sketch, batch_size=batch_size)[0][0:nb_img]
    for i in range(nb_img):
        plt.subplot(4, 4, i + 1)
        if i % 2 == 0:
            plt.imshow(img_sketch[i].squeeze(), cmap='Greys_r')
        else:
            plt.imshow(img_gen[i-1])
        plt.axis('off')
        # plt.subplot(nb_img, 2, i * 2 + 1)
        # plt.imshow(img_sketch[i].reshape((img_size,img_size)), cmap='Greys_r')
        #
        # plt.subplot(nb_img, 2, i * 2 + 2)
        # plt.imshow(img_gen[i])
        # plt.axis('off')
    plt.savefig("figures/%s.png" % tag)
    plt.clf()
    plt.close()

