require "numo/narray"
require_relative "functions.rb"

class Relu
  def initialize
    @mask = nil
  end

  def forward(x)
    @mask = (x <= 0)
    out = x.clone
    out[@mask.where] = 0

    return out
  end

  def backward(dout)
    dout[@mask.where] = 0
    dx = dout

    return dx
  end
end

class Sigmoid
  def initialize
    @out = nil
  end

  def forward(x)
    out = sigmoid(x)
    @out = out
    return out
  end

  def backward(dout)
    dx = dout * (1.0 - @out) * @out

    return dx
  end
end

class Affine
  attr_accessor :w, :b, :dw, :db

  def initialize(w, b)
    @w = w
    @b = b

    @x = nil
    @original_x_shape = nil
    # 重み・バイアスパラメータの微分
    @dw = nil
    @db = nil
  end

  def forward(x)
    # テンソル対応
    @original_x_shape = x.shape
    x = x.reshape(x.shape[0], x.shape.inject(:*) / x.shape[0])
    @x = x

    out = @x.dot(@w) + @b

    return out
  end

  def backward(dout)
    dx = dout.dot(@w.transpose())
    @dw = @x.transpose().dot(dout)
    @db = dout.sum(axis: 0)

    dx = dx.reshape(*@original_x_shape) # 入力データの形状に戻す（テンソル対応）
    return dx
  end
end

class SoftmaxWithLoss
  def initialize
    @loss = nil
    @y = nil # softmaxの出力
    @t = nil # 教師データ
  end

  def forward(x, t)
    @t = t
    @y = softmax(x)
    @loss = cross_entropy_error(@y, @t)

    return @loss
  end

  def backward(dout=1)
    batch_size = @t.shape[0]
    if @t.size == @y.size
      dx = (@y - @t) / batch_size
    else
      dx = @y.clone
      dx[0...batch_size, Numo::Bit.cast(@t).where] -= 1
      dx = dx / batch_size
    end

    return dx
  end
end
