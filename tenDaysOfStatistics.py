import math


class Solution:
    def __init__(self, X, W=None, do_sort=False):
        self.X = X
        self.do_sorted = do_sort
        if do_sort:
            self.X = sorted(X)
        self.W = W
        if W is None:
            self.W = [1 for _ in X]

    def get_mean(self) -> float:
        mean = 0.
        weights = 0.
        for i, x in enumerate(self.X):
            mean += self.W[i] * x
            weights += self.W[i]
        mean /= weights
        return mean

    def get_std_dev(self):
        sum = 0.
        the_mean = self.get_mean()
        for x in self.X:
            x -= the_mean
            x **= 2
            sum += x
        std_dev = math.sqrt(sum / len(self.X))
        return std_dev

    @staticmethod
    def calculate_median(sorted_list):
        #print(sorted_list)
        n = len(sorted_list)
        if len(sorted_list) % 2 != 0:
            return sorted_list[(n+1) // 2 - 1]
        else:
            return (sorted_list[n // 2 - 1] + sorted_list[n // 2]) / 2.

    def get_median(self):
        if not self.do_sorted:
            raise ValueError('do_sorted should be True')
        return self.calculate_median(self.X)

    def get_q1(self):
        if not self.do_sorted:
            raise ValueError('do_sorted should be True')
        return self.calculate_median(self.X[0:len(X) // 2])

    def get_q3(self):
        if not self.do_sorted:
            raise ValueError('do_sorted should be True')
        index = len(X) // 2
        if len(X) % 2 == 0:
            return self.calculate_median(self.X[index:])
        else:
            return self.calculate_median(self.X[index + 1:])



if __name__ == '__main__':
    n = int(input())
    X = [int(x) for x in input().split()]
    # W = [int(x) for x in input().split()]
    if len(X) != n:  # or len(W) != n:
        raise ValueError('N must match length of vector')
    solution = Solution(X, do_sort=True)
    result = solution.get_q1()
    print('{:.0f}'.format(result))
    result = solution.get_median()
    print('{:.0f}'.format(result))
    result = solution.get_q3()
    print('{:.0f}'.format(result))
