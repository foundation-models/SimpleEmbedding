import math
from typing import List


class Solution:
    weights = ...  # type: List[int]
    numbers = ...  # type: List[int]

    def __init__(self, numbers: List, frequencies: List = None, weights: List = None, do_sort: bool = False):
        """

        :type numbers: List[int]
        """
        self.numbers = numbers
        self.do_sorted = do_sort
        self.weights = weights
        self.frequency_map = {}
        if frequencies is not None:
            for i, x in enumerate(numbers):
                self.frequency_map[x] = self.frequency_map.get(x, 0) + frequencies[i]
        if weights is None:
            self.weights = [1 for _ in numbers]
        if do_sort:
            if self.frequency_map is None:
                self.numbers = sorted(numbers)
            else:
                self.numbers = []
                for x in sorted(list(set(numbers))):
                    f = self.frequency_map.get(x, 0)
                    if f > 0:
                        self.numbers.extend([x for _ in range(f)])
        #print(self.frequency_map)

    def get_mean(self) -> float:
        _mean = 0.
        _weights = 0.
        for i, x in enumerate(self.numbers):  # type: (int, int)
            _mean += self.weights[i] * x
            _weights += self.weights[i]
        _mean /= _weights
        return _mean

    def get_std_dev(self):
        _sum = 0.
        the_mean = self.get_mean()
        for x in self.numbers:
            x -= the_mean
            x **= 2
            _sum += x
        std_dev = math.sqrt(_sum / len(self.numbers))
        return std_dev

    @staticmethod
    def calculate_median(sorted_list):
        # print(sorted_list)
        n = len(sorted_list)
        if len(sorted_list) % 2 != 0:
            return sorted_list[(n + 1) // 2 - 1]
        else:
            return (sorted_list[n // 2 - 1] + sorted_list[n // 2]) / 2.

    def get_median(self):
        if not self.do_sorted:
            raise ValueError('do_sorted should be True')
        return self.calculate_median(self.numbers)

    def get_q1(self):
        if not self.do_sorted:
            raise ValueError('do_sorted should be True')
        return self.calculate_median(self.numbers[0:len(self.numbers) // 2])

    def get_q3(self):
        if not self.do_sorted:
            raise ValueError('do_sorted should be True')
        index = len(self.numbers) // 2
        if len(self.numbers) % 2 == 0:
            return self.calculate_median(self.numbers[index:])
        else:
            return self.calculate_median(self.numbers[index + 1:])

    def get_interquartile_range(self):
        return self.get_q3() - self.get_q1()


if __name__ == '__main__':
    n = int(input())
    X = [int(x) for x in input().split()]  # type: List[int]
    # W = [int(x) for x in input().split()]
    F = [int(x) for x in input().split()]
    if len(X) != n:  # or len(W) != n:
        raise ValueError('First input must match length of array (2nd input)')
    solution = Solution(numbers=X, frequencies=F, do_sort=True)
    print('{:.1f}'.format(solution.get_interquartile_range()))
    """
    check for 
20
10 40 30 50 20 10 40 30 50 20 1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10 10 40 30 50 20 10 40 30 50 20
answer should be 5.0
30
10 40 30 50 20 10 40 30 50 20 1 2 3 4 5 6 7 8 9 10 20 10 40 30 50 20 10 40 30 50
1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 10 40 30 50 20 10 40 30 50 20
answer should be 30.0
    """
    """
    result = solution.get_q1()
    print('{:.0f}'.format(result))
    result = solution.get_median()
    print('{:.0f}'.format(result))
    result = solution.get_q3()
    print('{:.0f}'.format(result))
    """
