from collections import Counter
def minWindow(s: str, t: str) -> str:
    ans = len(s)
    result = ""
    cnt = Counter()
    for x in t:
        cnt[x] += 1
    left = 0
    for right, c in enumerate(s):
        if c in cnt.keys():
            cnt[c] -= 1
        while all(v<=0 for v in cnt.values()):
            if right-left < ans:
                ans = right-left
                result = s[left:right+1]
            if s[left] in cnt.keys():
                cnt[s[left]] += 1
            left += 1
    return result

print(minWindow("ADOBECODEBANC", "ABC"))  # Expected output: "BANC"