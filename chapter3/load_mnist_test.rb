# encoding: utf-8
require "../dataset/mnist"
require "numo/gnuplot"

(x_train, y_train), * = MNIST.load_mnist(normalize: false)

print "x_train.shape: "
p x_train.shape
img = x_train[0, 0...784].reshape(28, 28)
label = y_train[0]
print "label: "
puts label
print "img.shape: "
p img.shape

Numo.gnuplot do
  set xrange: -10..38
  set yrange: 38..-10
  set :tics, :out
  set :palette, "gray"
  unset :colorbox
  plot img, with:"image"
end

print "終了: Enter"
_ = gets

