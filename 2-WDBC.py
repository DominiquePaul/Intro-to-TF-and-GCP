import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np

tf.reset_default_graph()

input_nodes = train_X.shape[1]

x = tf.placeholder(tf.float32, shape=[None, 30], name="input_placeholder")
y_ = tf.placeholder(tf.float32, shape=[None, 2], name="label")

W1 = tf.Variable(tf.zeros([30,2]))
b1 = tf.Variable(tf.zeros(2))

W2 =
b2 =

W3 =
b3 =

y_pred = tf.nn.softmax(tf.nn.relu(tf.matmul(x, W)+b))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_pred))

# add a softmax?
# softmax = tf.nn.softmax(logits=y_pred)
# loss = tf.reduce_sum(tf.square(tf.argmax(softmax, 1) - y_pred))
#loss = tf.reduce_sum(tf.square(y_ - y_pred, name='loss'))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)


# preparing data
raw_data = load_breast_cancer()
data = raw_data["data"]
targets = raw_data["target"]
# targets = np.where(targets==1,1,-1)
onehot_encoder = OneHotEncoder(categories="auto")
targets = onehot_encoder.fit_transform(targets.reshape(-1,1)).toarray()


x_train, x_test, y_train, y_test = train_test_split(data,targets, train_size=0.8, test_size=0.2)


def next_batch(x_data, y_data, size=64):
    idx = random.sample_without_replacement(x_data.shape[0], size)
    return(x_data[idx], y_data[idx])



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(20):
        x_batch, y_batch = next_batch(x_train, y_train)
        loss_train,_ = sess.run([loss, optimizer], feed_dict={x: x_batch, y_: y_batch})
        print(f"Batch no. {i}: loss = { loss_train }")

    print(f"final loss: {sess.run(loss, feed_dict={x: x_test, y_: y_test})}")
