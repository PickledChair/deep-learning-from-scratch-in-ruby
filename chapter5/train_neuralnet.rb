require "numo/narray"
require "charty"
require "../dataset/mnist"
require "./two_layer_net"

(x_train, t_train), (x_test, t_test) = \
  MNIST.load_mnist(normalize: true, one_hot_label: true)

network = TwoLayerNet.new(
  input_size: 784,
  hidden_size: 50,
  output_size: 10
)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_arr = []
train_acc_arr = []
test_acc_arr = []

iter_per_epoch = [train_size / batch_size, 1].max

iters_num.times do |i|
  batch_mask = batch_size.times.collect { rand(train_size) }
  x_batch = x_train[batch_mask, 0...784]
  t_batch = t_train[batch_mask, 0...10]

  # 誤差逆伝播法によって勾配を求める
  grad = network.gradient(x_batch, t_batch)

  # パラメータの更新
  [:W1, :b1, :W2, :b2].each do |key|
    # Numo::NArrayオブジェクトがAffineレイヤに値渡しされてしまっているらしいので、
    # network.paramsのパラメータを更新してもAffineレイヤのパラメータが更新されない。
    # 代わりに、姑息的にAffineレイヤのパラメータを直接更新した。
    # network.params[key] -= learning_rate * grad[key]
    if key == :W1
      network.layers[:Affine1].w -= learning_rate * grad[key]
    elsif key == :b1
      network.layers[:Affine1].b -= learning_rate * grad[key]
    elsif key == :W2
      network.layers[:Affine2].w -= learning_rate * grad[key]
    elsif key == :b2
      network.layers[:Affine2].b -= learning_rate * grad[key]
    end
  end

  loss = network.loss(x_batch, t_batch)
  train_loss_arr.push(loss)

  if i % iter_per_epoch == 0
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_arr.push(train_acc)
    test_acc_arr.push(test_acc)
    puts "train acc, test acc | " + train_acc.to_s + ", " + test_acc.to_s
  end
end

x = Numo::Int32.new(train_acc_arr.size).seq
charty = Charty::Plotter.new(:pyplot)
curve = charty.curve do
  series x, train_acc_arr, label: "train acc"
  series x, test_acc_arr, label: "test acc"
  xlabel "epochs"
  ylabel "accuracy"
  range y: 0..1
end

curve.render("train_neuralnet.png")
