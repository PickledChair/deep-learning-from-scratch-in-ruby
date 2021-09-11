require "../dataset/mnist"
require "./two_layer_net"

# データの読み込み
(x_train, t_train), (x_test, t_test) = \
  MNIST.load_mnist(normalize: true, one_hot_label: true)

network = TwoLayerNet.new(
  input_size: 784,
  hidden_size: 50,
  output_size: 10
)

x_batch = x_train[0...3, 0...784]
t_batch = t_train[0...3, 0...10]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

grad_numerical.each_key do |key|
  diff = (grad_backprop[key] - grad_numerical[key]).abs().mean
  puts key.to_s + ":" + diff.to_s
end
