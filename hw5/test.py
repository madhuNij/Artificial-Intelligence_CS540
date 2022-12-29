import collections
def getMinimumUniqueSum(arr):
    # Write your code here
        # Write your code here
    count = collections.Counter(arr)

    # array of unique values taken
    taken = []

    ans = 0

    for x in range(100000):
        if count[x] >= 2:
            taken.extend([x] * (count[x] - 1))
        elif taken and count[x] == 0:
            ans += x - taken.pop()

    print(ans)
arr = [2,2,2,2,2]
getMinimumUniqueSum(arr)