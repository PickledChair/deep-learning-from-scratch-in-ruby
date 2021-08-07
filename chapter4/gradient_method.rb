require "numo/narray"
require "charty"
require "../common/gradient.rb"

def gradient_descent(f, init_x, lr: 0.0, step_num: 100)
  x = init_x
  x_history = Array.new

  step_num.times { |i|
    x_history.push x.clone

    grad = GRAD.numerical_gradient(f, x)
    x -= lr * grad
  }

  return x, Numo::DFloat.cast(x_history)
end

def function_2(x)
  return x[0]**2 + x[1]**2
end

if $0 == __FILE__
  init_x = Numo::DFloat[-3.0, 4.0]

  lr = 0.1
  step_num = 20
  x, x_history = gradient_descent(method(:function_2), init_x, lr:lr, step_num:step_num)

  charty = Charty::Plotter.new(:pyplot)
  scatter = charty.scatter do
    series x_history[0...x_history.shape[0],0], x_history[0...x_history.shape[0],1]
    range x: -3.5..3.5, y: -4.5..4.5
    xlabel "X0"
    ylabel "X1"
  end

  scatter.render("gradient_method.png")
end
