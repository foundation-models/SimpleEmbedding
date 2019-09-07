def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def binomial_coefficient(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

def probability_mass_function(n, k, p):
    return binomial_coefficient(n,k) * p**k * (1 - p)**(n-k)

def cumulative_distribution_function(n, k, p):
    return sum([probability_mass_function(n, i, p) for i in range(k)])

def geometric_distribution(n,p):
    return (1-p)**(n-1) * p

def cumulative_geometric_distribution(n,p):
    return 1-(1-p)**(n+1)


#x, n = map(int, input().split())
#p = x / 100
"""
https://www.hackerrank.com/challenges/s10-binomial-distribution-1/problem
expected
0.696
"""
#print('{:0.3f}'.format(1 - cumulative_distribution_function(6,3,(1.09/2.09))))

"""
 https://www.hackerrank.com/challenges/s10-binomial-distribution-2/problem?h_r=next-challenge&h_v=zen
expected 
0.891
0.342
"""
#print('{:0.3f}'.format(cumulative_distribution_function(n,3,p)))
#print('{:0.3f}'.format(1 - cumulative_distribution_function(n,2,p)))

"""
https://www.hackerrank.com/challenges/s10-geometric-distribution-1/problem?h_r=next-challenge&h_v=zen
expected
0.868
"""
a, b = map(int, input().split())
n = int(input())
p = a / b
print('{:0.3f}'.format(cumulative_geometric_distribution(n-1,p)))
