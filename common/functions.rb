require "numo/narray"
NM = Numo::NMath

def identity_function(x)
  return x
end

def step_function(x)
  return Numo::Int32.cast(x > 0)
end

def sigmoid(x)
  return 1 / (1 + NM.exp(-x))
end

def sigmoid_grad(x)
  return (1.0 - sigmoid(x)) * sigmoid(x)
end

def relu(x)
  return x.clip(0, nil)
end

def relu_grad(x)
  grad = x.new_zeros
  grad[(x>=0).where] = 1
  return grad
end

def softmax(x)
  if x.shape.size > 1
    x = x - x.max(axis: 1, keepdims: true)
    return NM.exp(x) / NM.exp(x).sum(axis: 1, keepdims: true)
  else
    x = x - x.max
    return NM.exp(x) / NM.exp(x).sum
  end
end

def sum_squared_error(y, t)
  0.5 * ((y-t)**2).sum
end

def cross_entropy_error(y, t)
  if y.ndim == 1
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  end

  # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
  if t.size == y.size
    t = t.argmax(axis: 1)
  end

  batch_size = y.shape[0]
  y_ = Numo::DFloat.zeros(batch_size, 1)
  (0...batch_size).each { |i|
    y_[i, 0] = y[i, t[i]]
  }
  return -NM.log(y_ + 1e-7).sum / batch_size
end
