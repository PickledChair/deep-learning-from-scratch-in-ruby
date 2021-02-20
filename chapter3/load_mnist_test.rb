# encoding: utf-8
require "../dataset/mnist"
require "numo/gnuplot"

(x_train, t_train), (x_test, t_test) = MNIST.load_mnist(normalize: false)


print "x_train.shape: "
p x_train.shape
print "t_train.shape: "
p t_train.shape
print "x_test.shape: "
p x_test.shape
print "t_test.shape: "
p t_test.shape
label = t_train[0]
print "\nlabel: "
puts label
img = x_train[0, 0...784].reshape(28, 28)
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

