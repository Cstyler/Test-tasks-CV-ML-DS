#
# Input: nums = [2,7,11,15], target = 9
# Output: [0,1]
# Output: Because nums[0] + nums[1] == 9, we return [0, 1].
#
# Example 2:
#
# Input: nums = [3,2,4], target = 6
# Output: [1,2]
#
# Example 3:
#
# Input: nums = [3,3], target = 6
# Output: [0,1]

def f(nums, target):
    n = len(nums)
    for i in range(n - 1):
        for k in range(1, n - i - 1):
            r = nums[i] + nums[i + k]
            if r == target:
                return i, k
        else:
            r = nums[i] + nums[i + 1]
            if r == target:
                return i, i + 1


print(f([3, 3], 6))
print(f([3, 2, 4], 6))
print(f([2, 7, 11, 15], 9))
print(f([1, 1, 1, 1, 3, 2, 4], 6))