n = int(input())
arr = map(int, input().split())
first_max = 0
for i in arr:
    first_max = max(i, first_max)
ans = 0
for j in arr:
    ans = max(j, ans)