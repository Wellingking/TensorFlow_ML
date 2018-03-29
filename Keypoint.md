# Keypoint of TensorFlow relevant
# Photo pix memory

train_features Shape: (55000, 784) Type: float32   
how many bytes of memory does train_features need?   
172480000   
method: (55000 * 784 * 32 / 8)

# Problem 1 - normalize scale
The first problem involves normalizing the features for your training and test data.
Implement Min-Max scaling in the `normalize()` function to a range of `a=0.1` and `b=0.9`. After scaling, the values of the pixels in the input data should range from 0.1 to 0.9.
Since the raw notMNIST image data is in [grayscale](https://en.wikipedia.org/wiki/Grayscale), the current values range from a min of 0 to a max of 255.
`
Min-Max Scaling:
$
X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}
$
`
Implement the Min-Max scaling function ( X′=a+(X−Xmin)(b−a)Xmax−XminX′=a+(X−Xmin)(b−a)Xmax−Xmin ) with the parameters:

- Xmin=0  
- Xmax=255  
- a=0.1  
- b=0.9  

# Problem 2 - Set the features and labels tensors
```
features = tf.placeholder(tf.float32)   
labels = tf.placeholder(tf.float32)
```

# Problem 3 - Set the weights and biases tensors
```
weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))   
biases = tf.Variable(tf.zeros(labels_count))
```
# Test Cases   
```python
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

assert features._shape == None or (\
    features._shape.dims[0].value is None and\
    features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels._shape  == None or (\
    labels._shape.dims[0].value is None and\
    labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
assert weights._variable._shape == (784, 10), 'The shape of weights is incorrect'
assert biases._variable._shape == (10), 'The shape of biases is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'
```

# Feed dicts for training, validation, and test session
```
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}
```
# Linear Function WX + b

`logits = tf.matmul(features, weights) + biases`
`prediction = tf.nn.softmax(logits)`

# Cross entropy
`cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)`

1. some students have encountered challenges using this function, and have resolved issues
2. using https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
3. please see this thread for more detail https://discussions.udacity.com/t/accuracy-0-10-in-the-intro-to-tensorflow-lab/272469/9

# Training loss
`loss = tf.reduce_mean(cross_entropy)`

# Create an operation that initializes all variables
`init = tf.global_variables_initializer()`

# Test Cases   

```python
with tf.Session() as session:
    session.run(init)
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    session.run(loss, feed_dict=test_feed_dict)
    biases_data = session.run(biases)

assert not np.count_nonzero(biases_data), 'biases must be zeros'
print('Tests Passed!')
```
# Deep Neural Networks

Total number of parameters  

= size of W + size of b

= 28x28x10 + 10

= 7850

General: N(input) K(output) :
`(N+1) * K (parameters)`

# Convolution Neural Networks   
## Parameters of CNN   
Dimensionality
From what we've learned so far, how can we calculate the number of neurons of each layer in our CNN?

Given:
our input layer has a width of W and a height of H  
our convolutional layer has a filter size F  
we have a stride of S  
a padding of P  
and the number of filters K,  
the following formula gives us the width of the next layer: W_out =[ (W−F+2P)/S] + 1.  

The output height would be H_out = [(H-F+2P)/S] + 1.  

And the output depth would be equal to the number of filters D_out = K.  

The output volume would be W_out * H_out * D_out.  

- new_height = (input_height - filter_height + 2 * P)/S + 1
- new_width = (input_width - filter_width + 2 * P)/S + 1

Knowing the dimensionality of each additional layer helps us understand how large our model is and how our decisions around filter size and stride affect the size of our network.  

```
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'SAME'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```

# In summary TensorFlow uses the following equation for 'SAME' vs 'VALID'  

SAME Padding, the output height and width are computed as:   
```
>out_height = ceil(float(in_height) / float(strides[1]))  
>out_width = ceil(float(in_width) / float(strides[2]))  

#VALID Padding, the output height and width are computed as:  

>out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
>out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
```



# Total number of parameters   
```
Setup:   
H = height, W = width, D = depth   
We have an input of shape 32x32x3 (HxWxD)  
20 filters of shape 8x8x3 (HxWxD)  
A stride of 2 for both the height and width (S)  
Zero padding of size 1 (P)  
Output Layer  
14x14x20 (HxWxD)  
Nice job! :-)
That's right, there are 756560 total parameters. That's a HUGE amount! Here's how we calculate it:   
(8 * 8 * 3 + 1) * (14 * 14 * 20) = 756560   
8 * 8 * 3 is the number of weights, we add 1 for the bias. Remember, each weight is assigned to every single part of the output (14 * 14 * 20). So we multiply these two numbers together and we get the final answer.
```
---   
## Parameter Sharing:  
Nice job! :-)
That's right, there are 3860 total parameters. That's 196 times fewer parameters! Here's how the answer is calculated:  

`(8 * 8 * 3 + 1) * 20 = 3840 + 20 = 3860`

That's 3840 weights and 20 biases. This should look similar to the answer from the previous quiz. The difference being it's just 20 instead of (14 * 14 * 20). Remember, with weight sharing we use the same filter for an entire depth slice. Because of this we can get rid of 14 * 14 and be left with only 20.

## Pooling Mechanics
1. new_height = (input_height - filter_height)/S + 1  
2. new_width = (input_width - filter_width)/S + 1  

### Here's the corresponding code:  
```python
input = tf.placeholder(tf.float32, (None, 4, 4, 5))
filter_shape = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
padding = 'VALID'
pool = tf.nn.max_pool(input, filter_shape, strides, padding)  
```
# Two Variables: weights and bias name solution   
```python
tf.reset_default_graph()  # Remove the previous weights and bias  
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')  
weights = tf.Variable(tf.truncated_normal([2, 3]) ,name='weights_0')
```
# Save and Load   
```python
saver = tf.train.Saver()   
saver.save(sess, save_file)  
tf.reset_default_graph() # Remove the previous weights and bias  
saver.restore(sess, save_file) # Load the weights and bias - No Error      
```
# TensorFlow Convolution Layer
```python
"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).
"""
import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)

def conv2d(input):
    # Filter (weights and bias)
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    # TODO: Define the filter weights `F_W` and filter bias `F_b`.
    # NOTE: Remember to wrap them in `tf.Variable`, they are trainable parameters after all.
    F_W = tf.Variable(tf.truncated_normal((2, 2, 1, 3)))
    F_b = tf.Variable(tf.zeros(3))
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    strides = [1, 2, 2, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = 'VALID'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

out = conv2d(X)
'''



