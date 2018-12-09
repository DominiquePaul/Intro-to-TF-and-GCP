#importing tensorflow
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Data is stored in tensors. We can create tensors like this:
a = tf.constant([1],name='Const1')
b = tf.constant([2], name="Const2")

# We can add them together like this
c = tf.add(a,b)

# But printing the result gives us an unexpected result:
print(c)

# This is because TF is lazily evaluated. We have only created the
# graph of our application. We have to explicitly tell TF to evaluate
# the values. We do this by running them in a Tensorflow session.
with tf.Session() as sess:
    print(f"The result of a and b is { sess.run(c)[0] }")


# We can also feed in a numpy array to a tensor
img = cv2.imread("Data/sample_image1.png")
type(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# cv2.imshow()
d = tf.constant(img)

# Tensors can have any dimensions. This is useful if we want to process an
# image for example, which would have three dimensions: height, width and RGB
# colour.
print(d)
# If we would input a group of images, this would add a forth dimension
