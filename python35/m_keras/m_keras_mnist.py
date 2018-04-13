import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Dropout, Flatten, MaxPooling2D
from keras.datasets import mnist


# size setting
batch_size=50
num_classes=10
epochs=1

# input image size
img_rows,img_cols=28,28

# load image
(X_train, y_train), (X_test, y_test)=mnist.load_data()

# channels_last
X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test=X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
input_shape=(img_rows,img_cols,1)

# make the data smaller
X_train=X_train.astype("float32")
X_test=X_test.astype("float32")
X_train/=255
X_test/=255
y_train=keras.utils.np_utils.to_categorical(y_train,num_classes)
y_test=keras.utils.np_utils.to_categorical(y_test,num_classes)


# model
model=Sequential()

model.add(Convolution2D(32, input_shape=input_shape,nb_row=5,nb_col=5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,nb_row=5,nb_col=5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# compile
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

# fit
model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_test,y_test),verbose=1)

# evaluate
score=model.evaluate(X_test,y_test,verbose=0)
print("Test loss:",score[0])
print("Test accuracy:",score[1])