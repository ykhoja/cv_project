import tensorflow as tf

# Test if it imported TensorFlow properly
a = tf.constant(4)
b = tf.constant(5)
c = a + b
print(c)

with tf.Session() as sess:
	out = sess.run(c)
	print(out)