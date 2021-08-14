require "./layer_naive.rb"

apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer.new
mul_tax_layer = MulLayer.new

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

p price # 220

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

puts "#{dapple}, #{dapple_num}, #{dtax}" # 2.2 110 200
