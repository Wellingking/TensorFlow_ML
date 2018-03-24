# Keypoint of TensorFlow relevant
## photo pix memory

train_features Shape: (55000, 784) Type: float32   
how many bytes of memory does train_features need?   
172480000   
method: (55000 * 784 * 32 / 8)

# problem 1 - normalize scale
The first problem involves normalizing the features for your training and test data.
Implement Min-Max scaling in the `normalize()` function to a range of `a=0.1` and `b=0.9`. After scaling, the values of the pixels in the input data should range from 0.1 to 0.9.
Since the raw notMNIST image data is in [grayscale](https://en.wikipedia.org/wiki/Grayscale), the current values range from a min of 0 to a max of 255.

Min-Max Scaling:
$
X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}
$

Implement the Min-Max scaling function ( X′=a+(X−Xmin)(b−a)Xmax−XminX′=a+(X−Xmin)(b−a)Xmax−Xmin ) with the parameters:

Xmin=0  
Xmax=255  
a=0.1  
b=0.9  

# Problem 2 - Set the features and labels tensors
features = tf.placeholder(tf.float32)   
labels = tf.placeholder(tf.float32)

# Problem 2 - Set the weights and biases tensors
weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))   
biases = tf.Variable(tf.zeros(labels_count))
