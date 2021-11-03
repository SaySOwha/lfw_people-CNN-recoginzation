import ssl
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,BatchNormalization,Dense, Flatten,Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical

ssl._create_default_https_context = ssl._create_unverified_context
people = fetch_lfw_people(min_faces_per_person=70)
images= people.images
labels=people.target
n=7
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.20)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

img_x, img_y = 62,47
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu',
                 kernel_initializer='uniform',input_shape=(img_x, img_y, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'
                 ,kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'
                 ,kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'
                 ,kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120, activation='relu',kernel_initializer='uniform'))
model.add(Dense(84, activation='relu',kernel_initializer='uniform'))
model.add(Dense(n, activation='softmax'))

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y_train = to_categorical(y_train,n)
y_test = to_categorical(y_test,n)
# 7. 训练
model.fit(x_train, y_train,validation_split=0.20, validation_data=None, shuffle=True, batch_size=128, epochs=500)
score = model.evaluate(x_train, y_train)
print('training accuracy', score[1]*100,"%")
score = model.evaluate(x_test, y_test)
print('testing accuracy', score[1]*100,"%")
model.save('my_model_lfw.h5')


