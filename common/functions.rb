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

def relu(x)
  return x.clip(0, nil)
end

def softmax(x)
  x = x - x.max
  return NM.exp(x) / NM.exp(x).sum
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
  return -NM.log(y[0...batch_size, t] + 1e-7).sum / batch_size
end
