from cmath import inf
from collections import Counter, defaultdict
from typing import List


def sumDivisibleByK(nums: List[int], k: int) -> int:
    ans = 0
    cnt = Counter(nums)
    for num, count in cnt.items():
        if count % k == 0:
            ans += num * count
    return ans


# print(sumDivisibleByK([1, 2, 2, 3, 3, 3, 3, 4], 2))
# print(sumDivisibleByK([1, 2, 3, 4, 5], 2))


class Solution:
    def longestBalanced(self, s: str) -> int:
        n = len(s)
        ans = 0

        for left in range(n):
            freq = [0] * 26
            dist_set = set()

            for right in range(left, n):
                idx = ord(s[right]) - ord("a")
                freq[idx] += 1
                dist_set.add(s[right])

                min_freq = inf
                max_freq = 0
                for ch in dist_set:
                    cnt = freq[ord(ch) - ord("a")]
                    if cnt < min_freq:
                        min_freq = cnt
                    if cnt > max_freq:
                        max_freq = cnt

                if min_freq == max_freq:
                    ans = max(ans, right - left + 1)

        return ans


sol = Solution()
print(sol.longestBalanced('abbac'))
print(sol.longestBalanced('aabcc'))
print(sol.longestBalanced('aba'))
