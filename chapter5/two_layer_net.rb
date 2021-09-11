require "numo/narray"
require "random_bell"
require "../common/layers.rb"
require "../common/gradient.rb"


class TwoLayerNet
  attr_accessor :params, :layers

  def initialize(input_size:, hidden_size:, output_size:, weight_init_std: 0.01)
    # 重みの初期化
    @params = {}
    bell = RandomBell.new
    r0 = Numo::DFloat.zeros(input_size, hidden_size).map { bell.rand }
    @params[:W1] = weight_init_std * r0
    @params[:b1] = Numo::DFloat.zeros(hidden_size)
    r1 = Numo::DFloat.zeros(hidden_size, output_size).map { bell.rand }
    @params[:W2] = weight_init_std * r1
    @params[:b2] = Numo::DFloat.zeros(output_size)

    # レイヤの生成
    @layers = {}
    @layers[:Affine1] = Affine.new(@params[:W1], @params[:b1])
    @layers[:Relu1] = Relu.new
    @layers[:Affine2] = Affine.new(@params[:W2], @params[:b2])

    @last_layer = SoftmaxWithLoss.new
  end

  def predict(x)
    @layers.each_value do |layer|
      x = layer.forward(x)
    end

    return x
  end

  # x: 入力データ, t: 教師データ
  def loss(x, t)
    y = self.predict(x)
    return @last_layer.forward(y, t)
  end

  def accuracy(x, t)
    y = self.predict(x)
    y = y.argmax(axis:1)
    if t.ndim != 1
      t = t.argmax(axis:1)
    end

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
    # forward
    self.loss(x, t)

    # backward
    dout = 1
    dout = @last_layer.backward(dout)

    layers = @layers.values.reverse
    # layers.reverse!
    layers.each do |layer|
      dout = layer.backward(dout)
    end

    # 設定
    grads = {}
    grads[:W1] = @layers[:Affine1].dw
    grads[:b1] = @layers[:Affine1].db
    grads[:W2] = @layers[:Affine2].dw
    grads[:b2] = @layers[:Affine2].db

    return grads
  end
end