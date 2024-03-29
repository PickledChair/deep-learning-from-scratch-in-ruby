require "numo/narray"

module GRAD
  class << self
    def numerical_gradient_1d(f, x)
      h = 1e-4 # 0.0001
      grad = x.new_zeros

      (0...x.size).each { |idx|
        tmp_val = x[idx]
        x[idx] = tmp_val.to_f + h
        fxh1 = f.call(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f.call(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
      }

      return grad
    end
    private :numerical_gradient_1d

    def numerical_gradient_2d(f, x_)
      if x_.ndim == 1
        return numerical_gradient_1d(f, x_)
      else
        grad = x_.new_zeros

        idx = 0
        x_.each_over_axis { |x|
          grad[idx, 0...x_.shape[1]] = numerical_gradient_1d(f, x)
          idx += 1
        }

        return grad
      end
    end

    def numerical_gradient(f, x)
      h = 1e-4 # 0.0001
      grad = x.new_zeros

      x.each_with_index { |num, *idx|
        tmp_val = num
        x[*idx] = tmp_val + h
        fxh1 = f.call(x) # f(x+h)

        x[*idx] = tmp_val - h
        fxh2 = f.call(x) # f(x-h)
        grad[*idx] = (fxh1 - fxh2) / (2*h)

        x[*idx] = tmp_val
      }

      return grad
    end
  end
end
