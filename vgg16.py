import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
import math
from  tensorflow.keras.datasets import mnist
import numpy as np
# tf.enable_eager_execution()
(train,train_label),(test, test_label) = mnist.load_data()
from PIL import Image

# print(train.shape)
# print(type(train))
# print(train_label.shape)
# print(type(train_label))

batch_size = 16
# train_path = "C:/Users/thang_dinh/Documents/個人データセット/Train_Cartoon"
# test_path = "C:/Users/thang_dinh/Documents/個人データセット/Test_Cartoon"

train_path = "./Train_Cartoon"
test_path = "./Test_Cartoon"
n_epochs = 1

# def convert_itr(data_itr):
#
#     for i in range(len(data_itr)):
#         print(data_itr[i][1])
#         data_itr[i][1] = np.int32(data_itr[i][1])
#
#     # for exam in data_itr:
#     #     train_ = exam[0]
#     #     label_ = exam[1]
#     #     x.append(train_)
#     #     y.append(label_)
#     # x = np.asarray(x)
#     # y = np.asarray(y)
#     return data_itr


vgg16 = VGG16(include_top=False, input_shape=(224,224,3))


def loss_function(logits, pred):
    print("test loss function")
    print("Pred shape", pred.shape)
    print("Logits shape", logits.shape)
    logits = tf.nn.softmax(logits)
    loss = -tf.reduce_mean(tf.multiply(tf.log(logits), pred))
    return loss

idg_train = ImageDataGenerator(
    rescale=1/255.,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

img_itr_train = idg_train.flow_from_directory(train_path, target_size=(224,224), batch_size=batch_size,class_mode="categorical" )
img_itr_validation = idg_train.flow_from_directory(test_path, target_size=(224,224), batch_size=batch_size, class_mode="categorical")
# x_train, y_train = convert_itr(img_itr_train)

print(img_itr_train[0][1])
print(img_itr_train[0][1].shape)

step_per_epoch = math.ceil(img_itr_train.samples/batch_size)
validation_steps = math.ceil(img_itr_validation.samples/batch_size)

def build_transfer_model(vgg16):
    model = Sequential(vgg16.layers)
    for layer in model.layers[:15]:
        layer.trainable = False
    model.add(Flatten())
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation=tf.nn.sigmoid))
    return model


model = build_transfer_model(vgg16)
model.compile(loss=loss_function, optimizer=SGD(lr=1e-4, momentum=0.9), metrics = ["accuracy"])

model.fit_generator(img_itr_train, steps_per_epoch=step_per_epoch, epochs = n_epochs, validation_data=img_itr_validation, validation_steps=validation_steps )
img_test = img_itr_validation[0][0]
result = model.predict(img_test)
label = img_itr_validation[0][1]

print(np.argmax(result, axis = 1))
print(np.argmax(label, axis = 1))

# print(img_test.shape)
# print(img_itr_validation[0][1].shape)
for index in range(len(img_test)):
    # print(img_test[index])
    img = array_to_img(img_test[index])
    img.save(str(index)+".jpg")
