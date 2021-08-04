require '../dataset/mnist'
require '../common/functions'

def get_data
  _, (x_test, t_test) = MNIST.load_mnist(normalize: true,
                                         flatten: true,
                                         one_hot_label: false)
  return x_test, t_test
end

def init_network
  data = File.binread("sample_weight.dump")
  network = Marshal.load(data)
  return network
end

def predict(network, x)
  w1, w2, w3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = x.dot(w1) + b1
  z1 = sigmoid(a1)

  a2 = z1.dot(w2) + b2
  z2 = sigmoid(a2)

  a3 = z2.dot(w3) + b3
  y = softmax(a3)

  return y
end

if $0 == __FILE__
  x, t = get_data
  network = init_network

  batch_size = 100
  accuracy_cnt = 0

  0.step(by: batch_size, to: x.shape[0] - batch_size) { |i|
    x_batch = x[i...(i+batch_size), 0...784]
    y_batch = predict(network, x_batch)
    p = y_batch.argmax(axis: 1)
    accuracy_cnt += p.eq(t[i...(i+batch_size)]).count_true
  }

  puts "Accuracy: " + (accuracy_cnt.to_f / x.shape[0].to_f).to_s
end
