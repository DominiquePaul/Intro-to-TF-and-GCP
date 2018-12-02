import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(30), activation=tf.nn.relu)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])

# preparing data
raw_data = load_breast_cancer()
data = raw_data["data"]
targets = raw_data["target"]
# onehot_encoder = OneHotEncoder(categories="auto")
# targets = onehot_encoder.fit_transform(targets.reshape(-1,1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(data,targets, train_size=0.8, test_size=0.2)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model.fit(x_train, y_train, epochs=50)  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy
