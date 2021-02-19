require "magro"
require "../dataset/mnist"

(x_train, y_train), * = MNIST.load_mnist(normalize: false)

img = x_train[0, 0...784].reshape(28, 28)
label = y_train[0]
print "label: "
p label
print "image shape: "
p img.shape

Magro::IO.imsave("mnist_five.png", img)

