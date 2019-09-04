def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def binomial_coefficient(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

def probability_mass_function(n, k, p):
    return binomial_coefficient(n,k) * p**k * (1 - p)**(n-k)

def cumulative_distribution_function(n, k, p):
    return sum([probability_mass_function(n, i, p) for i in range(k)])

print('{:0.3f}'.format(1 - cumulative_distribution_function(6,3,(1.09/2.09))))