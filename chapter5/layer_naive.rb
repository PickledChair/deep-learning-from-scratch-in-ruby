class MulLayer
  def initialize
    @x = nil
    @y = nil
  end

  def forward(x, y)
    @x = x
    @y = y
    out = x * y

    return out
  end

  def backward(dout)
    dx = dout * @y # x と y をひっくり返す
    dy = dout * @x

    return dx, dy
  end
end

class AddLayer
  def forward(x, y)
    out = x + y
    return out
  end

  def backward(dout)
    dx = dout * 1
    dy = dout * 1
    return dx, dy
  end
end
