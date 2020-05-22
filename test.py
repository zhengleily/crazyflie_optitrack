L = [1,2,5,8,21,6,49,10,20,80,90,121]
b = []
c = []
for num in L:
    if num > 2:
        flag = True
        for i in range(2, int(num**0.5 + 1)):
            if num % i == 0:
                flag = False
        if flag:
            c.append(num)
        else:
            b.append(num)
    else:
        c.append(num)
print(b,c)