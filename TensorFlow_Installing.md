
Throughout this lesson, you'll apply your knowledge of neural networks on real datasets using TensorFlow (link for China), an open source Deep Learning library created by Google.

You’ll use TensorFlow to classify images from the notMNIST dataset - a dataset of images of English letters from A to J. You can see a few example images below.


Your goal is to automatically detect the letter based on the image in the dataset. You’ll be working on your own computer for this lab, so, first things first, install TensorFlow!

## Install
OS X, Linux, Windows
Prerequisites
Intro to TensorFlow requires Python 3.4 or higher and Anaconda. If you don't meet all of these requirements, please install the appropriate package(s).

## Install TensorFlow
You're going to use an Anaconda environment for this class. If you're unfamiliar with Anaconda environments, check out the official documentation. More information, tips, and troubleshooting for installing tensorflow on Windows can be found here.

Run the following commands to setup your environment:

```
conda create --name=IntroToTensorFlow python=3 anaconda
source activate IntroToTensorFlow
conda install -c conda-forge tensorflow
```

That's it! You have a working environment with TensorFlow. Test it out with the code in the Hello, world! section below.

Docker on Windows
Docker instructions were offered prior to the availability of a stable Windows installation via pip or Anaconda. Please try Anaconda first, Docker instructions have been retained as an alternative to an installation via Anaconda.

Install Docker
Download and install Docker from the official Docker website.

Run the Docker Container
Run the command below to start a jupyter notebook server with TensorFlow:

docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow
Users in China should use the b.gcr.io/tensorflow/tensorflow instead of gcr.io/tensorflow/tensorflow

You can access the jupyter notebook at localhost:8888. The server includes 3 examples of TensorFlow notebooks, but you can create a new notebook to test all your code.

Hello, world!
Try running the following code in your Python console to make sure you have TensorFlow properly installed. The console will print "Hello, world!" if TensorFlow is installed. Don’t worry about understanding what it does. You’ll learn about it in the next section.
```python
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```
*Errors*
If you're getting the error tensorflow.python.framework.errors.InvalidArgumentError: Placeholder:0 is both fed and fetched, you're running an older version of TensorFlow. Uninstall TensorFlow, and reinstall it using the instructions above. For more solutions, check out the Common Problems section.
