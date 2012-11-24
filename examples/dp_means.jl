load("Clustering")
using Clustering

function generate_data(n::Int64)
  data = zeros(n, 2)
  mu_x = [0.0, 5.0, 10.0, 15.0]
  mu_y = [0.0, 5.0, 0.0, -5.0]
  
  for i = 1:n
    assignment = randi(4)
    data[i, 1] = randn(1)[1] + mu_x[assignment]
    data[i, 2] = randn(1)[1] + mu_y[assignment]
  end
  
  data
end

generate_data() = generate_data(1000)

data = generate_data()
dp_means(data, 0.5)
