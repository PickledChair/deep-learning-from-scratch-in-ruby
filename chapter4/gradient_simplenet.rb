require "numo/narray"
require "random_bell"
require "../common/functions.rb"
require "../common/gradient.rb"

class SimpleNet
  attr_accessor :w

  def initialize
    @w = Numo::DFloat.zeros(2,3)
    bell = RandomBell.new
    @w = @w.map { bell.rand }
  end

  def predict(x)
    return x.dot(@w)
  end

  def loss(x, t)
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss
  end
end

if $0 == __FILE__
  x = Numo::DFloat[0.6,0.9]
  t = Numo::DFloat[0, 0, 1]

  net = SimpleNet.new

  f = proc { |w| net.loss(x, t) }
  dW = GRAD.numerical_gradient(f, net.w)

  p dW
end
