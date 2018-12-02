import tensorflow as tf


# create

a = tf.constant([1],name='Const1')

b = tf.constant([2], name="Const2")

c = tf.add(a,b)

with tf.Session() as sess:
    print("The sum is: {}".format(sess.run(c[0])))

d = tf.constant([10,10])

e = tf.subtract(d,c)

with tf.Session() as sess:
    print(sess.run(e))


print(a)
