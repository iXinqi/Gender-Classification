import os
import random
import cv2
import numpy as np
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.optimizers import SGD

IMAGE_SIZE1 = 178
IMAGE_SIZE2 = 218
# 训练图片大小
epochs = 3
# 遍历次数
batch_size = 64
# 批量大小
nb_train_samples = 157336
# 训练样本总数
nb_validation_samples = 17250
#验证样本总数
nb_test_samples = 28013
# 测试样本总数
train_data_dir = './part_data/train'
validation_data_dir = './part_data/valid'
test_data_dir = './part_data/test'
# 样本图片所在路径
FILE_PATH = 'Gender.h5'
# 模型存放路径
class Dataset(object):

    def __init__(self):
        self.train = None
        self.valid = None
        self.test = None


    def read(self, img_rows=IMAGE_SIZE1, img_cols=IMAGE_SIZE2):
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        valid_datagen = ImageDataGenerator(rescale=1. / 255)
        
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = valid_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary')
        
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary')
        

        self.train = train_generator
        self.valid = validation_generator
        self.test = test_generator


class Model(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE1,IMAGE_SIZE2,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))


    def train(self, dataset, batch_size=batch_size, nb_epoch=epochs):

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        self.model.fit_generator(dataset.train,
                                 steps_per_epoch=nb_train_samples // batch_size,
                                 epochs=epochs,
                                 validation_data=dataset.valid,
                                 validation_steps=nb_validation_samples//batch_size)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save_weights(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model.load_weights(file_path)
        print(file_path)

    def predict(self, image):
        # 预测样本分类
        img = image.resize((1, IMAGE_SIZE1, IMAGE_SIZE2, 3))
        img = image.astype('float32')
        img /= 255
        #归一化
        result = self.model.predict(img)
        print(result)
        # 概率
        result = self.model.predict_classes(img)
        print(result[0])
        # 0/1

        return result[0]
    
    def evaluate(self, dataset):
        # 测试样本准确率
        score = self.model.evaluate_generator(dataset.test,steps=10)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
    

if __name__ == '__main__':
    dataset = Dataset()
    dataset.read()

    model = Model()
#    model.load()
    model.train(dataset)
    model.evaluate(dataset)
    model.save()
    
    
