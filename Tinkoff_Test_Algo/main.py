
def memoizator(fun):
    d = {}
    def fib2(n):
        if n in d:
            return d[n]
        res = fib(n)
        d[n] = res
        return res
    return fib2

@memoizator
def fib(n: int) -> int:
    """
    calculate the fibonacci number in a recursive way
    reminder: fib(0) = fib(1) = 1, fib(2) = fib(1) + fib(0) = 2, etc
    :param n: 
    :return: n-th fibonacci number
    """
    assert n >= 0
    if n == 0 or n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)
 
if __name__ == '__main__':
    pass
