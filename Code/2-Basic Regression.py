# import Iris dataset
import tensorflow as tf
from sklearn import datasets
import numpy as np

# Hyperparameters
learning_rate = 0.02
iterations = 2000

# Defining Placeholders: These are tensors but without any values. We are
# telling TF to create tensors with the promise that we will later fill in
# values for these. This is good if our training data isn't always the same,
# for example when we are using batches.
x = tf.placeholder(tf.float32,shape=[None,1], name="input-data")
y = tf.placeholder(tf.float32, shape=[None,1], name="ground-truth-labels")

# Defining Variables: Tensorflow knows that it can adjust the weights of these
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Our prediction formula: y_hat = ß0 + ß1 * X1
y_pred = tf.add(tf.multiply(x,W), b)

# We use the MSE as a measure of our error.
loss = tf.reduce_sum(tf.pow(y_pred - y, 2)) / (2 * 100)

# Defining our optimizer. The optimizer is a TF object that takes care of
# of adjusting our model. It calculates the gradients and adjusts our weights.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


### Adding the Data ###
data = datasets.load_iris()
features = data["data"][:100,0].reshape(100,1)
labels = data["target"][:100].reshape(100,1)
s = np.arange(features.shape[0])
np.random.shuffle(s)
features = features[s]
labels = labels[s]

def calc_accuracy(reg_vals, true_vals, limit):
    reg_vals = np.array(reg_vals)
    preds = np.where(reg_vals > limit,1,0)
    acc = np.abs(preds - true_vals).sum()
    return(acc)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(iterations):

        # we run the session. Note how the feed_dict variable passes on
        # the variables for our placeholder tensors
        loss_train, _ = sess.run([loss, optimizer], feed_dict={x: features, y: labels})
        if i % 200 == 0:
            print(f"Iteration {i}, Loss: {loss_train}")

    loss_test = sess.run([loss], feed_dict={x:features, y:labels})
    print(f"Our final loss is {np.round(loss_test,3)}.")

    y_preds_final = sess.run([y_pred], feed_dict={x:features, y:labels})


calc_accuracy(y_preds_final, labels, 0.5)
