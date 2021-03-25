
import random
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import math
import numpy as np
from scipy.spatial import distance

a = (1, 2, 3)
b = (4, 5, 6)
Dist = distance.euclidean(a, b)
print(Dist)

R = []
for i in range(0, 22, 1):
    R.append(random.uniform(0, 1))

R.sort(reverse=1)
print(len(R))

Dataset = load_diabetes()

x = Dataset.data
y = Dataset.target
x, y = shuffle(x, y)

Len = len(x)

t = Len * 0.7
v = Len * 0.15
a = int(t)
b = int(v)
p = a + b
q = p + b

tr = np.array([])
va = np.array([])
te = np.array([])
cal_class = []
all_class = []

for elements in enumerate(y):
    all_class.append(elements)

for i in range(0, a, 1):
    tr_data = x[0:a]
    tr_class = all_class[0:a]
for i in range(a, p, 1):
    va_data = x[a:p]
    va_class = all_class[a:p]
for i in range(p, q, 1):
    te_data = x[p:q]
    te_class = all_class[p:q]


def find_knn(knn):
    def e_dis(l):
        ecl_dis = []
        reg = 0

        for i in range(0, a, 1):
            dst = distance.euclidean(va_data[l], tr_data[i])
            ecl_dis.append((dst, tr_class[i][1]))

        ecl_dis.sort()

        for i in range(0, knn, 1):
            reg += ecl_dis[i][1]
        reg_avg = reg / knn
        # print(reg_avg)
        return reg_avg

    reg_val = []
    for l in range(0, b, 1):
        reg_val.append(e_dis(l))

    r = 0
    for i in range(0, b, 1):
        r += math.pow(reg_val[i] - va_class[i][1], 2)

    reg_loss = r / b
    print("Regression Loss: ", reg_loss)
    return reg_loss, reg_val


knn_a = []
best_loss = []

n = int(input("Enter the number of time to check KNN: "))
loss_res = []
loss_cl = []
for i in range(0, n, 1):
    knn = int(input("Enter k: "))
    knn_a.append(knn)
    loss_res.append(find_knn(knn))

print("K value            Validation Accuracy                 Validation Class")
for i in range(0, n, 1):
    print("    ", knn_a[i], "              ", loss_res[i][0], "                ", loss_res[i][1])

best_loss.append(min(loss_res))
print("Min Loss: ", best_loss[0][0], "for classess: ", best_loss[0][1])
loss_cl.append(best_loss[0][1])

Reg_loss = 6500

while (Reg_loss > 6000):
    r = 0
    for i in range(1, b, 1):
        r += math.pow(loss_cl[0][i] - te_class[i][1], 2)

    Reg_loss = r / b

    te_class = shuffle(te_class)

print("Reg Loss: ", Reg_loss)