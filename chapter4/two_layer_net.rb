require "numo/narray"
require "random_bell"
require "../common/functions.rb"
require "../common/gradient.rb"

class TwoLayerNet
  attr_accessor :params

  def initialize(input_size:, hidden_size:, output_size:, weight_init_std: 0.01)
    @params = {}
    bell = RandomBell.new
    r0 = Numo::DFloat.zeros(input_size, hidden_size).map { bell.rand }
    @params[:W1] = weight_init_std * r0
    @params[:b1] = Numo::DFloat.zeros(hidden_size)
    r1 = Numo::DFloat.zeros(hidden_size, output_size).map { bell.rand }
    @params[:W2] = weight_init_std * r1
    @params[:b2] = Numo::DFloat.zeros(output_size)
  end

  def predict(x)
    w1, w2 = @params[:W1], @params[:W2]
    b1, b2 = @params[:b1], @params[:b2]

    a1 = x.dot(w1) + b1
    z1 = sigmoid(a1)
    a2 = z1.dot(w2) + b2
    y = softmax(a2)

    return y
  end

  # x: 入力データ, t: 教師データ
  def loss(x, t)
    y = self.predict(x)

    return cross_entropy_error(y, t)
  end

  def accuracy(x, t)
    y = self.predict(x)
    y = y.argmax(axis:1)
    t = t.argmax(axis:1)

    acc = y.eq(t).count_true / x.shape[0].to_f
    return acc
  end

  # x: 入力データ, t: 教師データ
  def numerical_gradient(x, t)
    loss_W = proc { |w| self.loss(x, t) }

    grads = {}
    grads[:W1] = GRAD.numerical_gradient(loss_W, @params[:W1])
    grads[:b1] = GRAD.numerical_gradient(loss_W, @params[:b1])
    grads[:W2] = GRAD.numerical_gradient(loss_W, @params[:W2])
    grads[:b2] = GRAD.numerical_gradient(loss_W, @params[:b2])

    return grads
  end

  def gradient(x, t)
    w1, w2 = @params[:W1], @params[:W2]
    b1, b2 = @params[:b1], @params[:b2]
    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = x.dot(w1) + b1
    z1 = sigmoid(a1)
    a2 = z1.dot(w2) + b2
    y = softmax(a2)

    # backword
    dy = (y - t) / batch_num
    grads[:W2] = z1.transpose.dot(dy)
    grads[:b2] = dy.sum(axis:0)

    dz1 = dy.dot(w2.transpose)
    da1 = sigmoid_grad(a1) * dz1
    grads[:W1] = x.transpose.dot(da1)
    grads[:b1] = da1.sum(axis:0)

    return grads
  end
end
