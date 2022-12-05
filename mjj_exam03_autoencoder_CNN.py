import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(28, 28, 1))        # 가로x세로x컬러 의 형태이기에 픽셀값(1 컬러)를 하나하나씩 묶어준다.
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)        # padding='same' 을 줬기 때문에 그대로! 16장이 된다.
x = MaxPool2D((2, 2), padding='same')(x)        # MaxPool을 거치게 되면 인풋 shape이 (14, 14)로 된다.
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)     # 16장에서 8장이 된다.
x = MaxPool2D((2, 2), padding='same')(x)        # 인풋 shape이 (7, 7)로 된다.
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)     # (7, 7)은 한칸이 남지만 padding='same'을 줬기 때문에 (8, 8)이 된다.
encoded = MaxPool2D((2, 2), padding='same')(x)      # 인풋 shape이 (4, 4)로 된다.

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)     # 해당되는 값으로 2배씩하여 채워준다. 인풋 shape이 (8, 8)으로 늘어난다.
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)     # (16, 16)
x = Conv2D(16, (3, 3), activation='relu')(x)      # padding을 뺀다. 16에서 2를 빼서 shape이 (14, 14)가 된다.
x = UpSampling2D((2, 2))(x)     # (28, 28)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)      # 마지막 출력이 0~1값이기에 sigmoid를 준다.

autoencoder = Model(input_img, decoded)
autoencoder.summary()


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()       # 답이 없는 것 (라벨 X) - 비지도 학습 / 타겟이 있으므로 자기지도학습이라고도 한다.
x_train = x_train / 255
x_test = x_test / 255
# print(x_train.shape)
# print(x_train[0])
conv_x_train = x_train.reshape(-1, 28, 28, 1)       # -1를 주는 이유? mnist 60000장을 묶기 위해 줌
conv_x_test = x_test.reshape(-1, 28, 28, 1)
# print(conv_x_train.shape)
# print(conv_x_train)
# exit()

# 데이터에 노이즈(잡음) 주기
noise_factor = 0.5
conv_x_train_noisy = conv_x_train + np.random.normal(0, 1, size=conv_x_train.shape) * noise_factor
conv_x_train_noisy = np.clip(conv_x_train_noisy, 0.0, 1.0)      # 데이터의 상한과 하한 제한을 준다. 0보다 작으면 0.0으로 1보다 크면 1.0으로 만들어준다.
conv_x_test_noisy = conv_x_test + np.random.normal(0, 1, size=conv_x_test.shape) * noise_factor
conv_x_test_noisy = np.clip(conv_x_test_noisy, 0.0, 1.0)
plt.figure(figsize=(20, 4))
n = 10
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

fit_hist = autoencoder.fit(conv_x_train_noisy, conv_x_train, epochs=50,
                        batch_size=256, validation_data=(conv_x_test_noisy, conv_x_test))
autoencoder.save('./models/autoencoder_noisy.h5')

decoded_img = autoencoder.predict(conv_x_test[:10])



plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()




