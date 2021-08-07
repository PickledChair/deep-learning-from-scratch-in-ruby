require "numo/narray"
require "charty"

def numerical_diff(f, x)
  h = 1e-4 # 0.0001
  return (f.call(x+h) - f.call(x-h)) / (2*h)
end

def function_1(x)
  return 0.01*x**2 + 0.1*x
end

def tangent_line(f, x)
  d = numerical_diff(f, x)
  p d
  y = f.call(x) - d*x
  return proc { |t| d*t + y }
end

if $0 == __FILE__
  x = Numo::DFloat.new((20.0/0.1).to_i).seq(0.0, 0.1)
  y = function_1(x)

  tf = tangent_line(method(:function_1), 5)
  y2 = tf.call(x)

  charty = Charty::Plotter.new(:pyplot)
  curve = charty.curve do
    series x, y, label: "function_1"
    series x, y2, label: "tangent_line"
    xlabel "x"
    ylabel "f(x)"
  end

  curve.render("gradient_1d.png")
end
