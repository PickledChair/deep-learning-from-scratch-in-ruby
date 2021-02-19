require "numo/narray"
require "charty"
NM = Numo::NMath

def sigmoid(x)
  return 1 / (1 + NM.exp(-x))
end


if $0 == __FILE__
  x = Numo::DFloat.new(101).seq(-5, 0.1)
  y = sigmoid(x)

  charty = Charty::Plotter.new(:pyplot)
  curve = charty.curve do
    series x, y, label: "sigmoid function"
    xlabel "x"
    ylabel "y"
  end

  curve.render("sigmoid_function.png")
end
