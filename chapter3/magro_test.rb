require "npy"
require "magro"

data = Npy.load_npz("mnist.npz")
x_train = data["x_train"]
y_train = data["y_train"]

img = x_train[0, 0..27, 0..27]
label = y_train[0]
puts "label:"
puts label
puts "image shape:"
puts img.shape

Magro::IO.imsave("mnist_five.png", img)

