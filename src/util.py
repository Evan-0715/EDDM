'''
Tool functions
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import image
from data_process import read_image, add_noise
from tensorflow.keras.layers import concatenate

subfig_scale = 64
scale = 512
subfig_num = (scale // subfig_scale) ** 2


def psnr_pred(y_true, y_pred):
    return image.psnr(y_true, y_pred, max_val=1.0)


def ssim_pred(y_true, y_pred):
    return image.ssim(y_true, y_pred, max_val=1.0)


def read_pics(DATA_SET, PIC_NUM, SIGMA):
    clean_pic = read_image('{}/{}.png'.format(DATA_SET, PIC_NUM))
    clean_pic1 = read_image('{}/{}.png'.format(DATA_SET, PIC_NUM))
    noise_pic = add_noise(clean_pic1, SIGMA)
    clean_pic = clean_pic / 255
    noise_pic = noise_pic / 255
    return clean_pic, noise_pic


def split_image(img):
    subfig_scale = 64
    scale = 512
    '''
    img: np.ndarray
    '''
    res = np.zeros((1, subfig_scale, subfig_scale, 3))
    for i in range(0, scale, subfig_scale):
        for j in range(0, scale, subfig_scale):
            crop = img[i: i + subfig_scale, j: j + subfig_scale, :]
            crop_new = crop[np.newaxis, :, :, :]
            if i == 0 and j == 0:
                res[:] = crop_new
            else:
                res = np.concatenate([res, crop_new], axis=0)
    return np.array(res)


def show_pic(model, clean_pic, noise_pic):
    combine_pic = concatenate([noise_pic, clean_pic], axis=-1)
    combine_pic_1 = combine_pic[np.newaxis, :, :, :]
    _, _, predict_unsqueeze = model.predict(combine_pic_1, batch_size=1, verbose=1)
    predict1 = predict_unsqueeze.squeeze()
    predict = np.clip(predict1, 0, 1)
    plt.subplot(1, 3, 1)
    plt.imshow(noise_pic)
    plt.title('Noisy Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(clean_pic)
    plt.title('Gt Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(predict)
    plt.title('Denoised Image')
    plt.axis('off')
    plt.show()


def save_pic(model, clean_pic, noise_pic, path):
    combine_pic = concatenate([noise_pic, clean_pic], axis=-1)
    combine_pic_1 = combine_pic[np.newaxis, :, :, :]
    _, _, predict_unsqueeze = model.predict(combine_pic_1, batch_size=1, verbose=1)
    predict_pic = predict_unsqueeze.squeeze()
    predict_pic = np.clip(predict_pic, 0, 1)
    plt.figure(figsize=(16, 10))
    plt.subplot(131), plt.imshow(clean_pic)
    plt.subplot(132), plt.imshow(noise_pic)
    plt.subplot(133), plt.imshow(predict_pic)
    plt.savefig(path)
    plt.close()
