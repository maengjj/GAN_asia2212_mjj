import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')          # input을 따로 안줌 위에 줌 # relu는 넘어간다.
encoded = encoded(input_img)
decoded = Dense(784, activation='sigmoid')      # 출력은 0~1 사이면 된다. # 위에 encoded를 뱉어낸다. # minmax정규화를 함
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()

encoder = Model(input_img, encoded)
encoder.summary()

encoder_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()       # 모델이 3개 있는것 처럼 보이지만 실제론 하나이다.

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()       # 답이 없는 것 (라벨 X) - 비지도 학습 / 타겟이 있으므로 자기지도학습이라고도 한다.
x_train = x_train / 255
x_test = x_test / 255

flatted_x_train = x_train.reshape(-1, 784)
flatted_x_test = x_test.reshape(-1, 784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50,
                        batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
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




