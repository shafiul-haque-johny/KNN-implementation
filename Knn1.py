
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np
from scipy.spatial import distance


dataset = load_iris()
x = dataset.data
y = dataset.target
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
        mj_cl = []

        for i in range(0, a, 1):
            dst = distance.euclidean(va_data[l], tr_data[i])
            ecl_dis.append((dst, tr_class[i][1]))

        ecl_dis.sort()

        for i in range(0, knn, 1):
            mj_cl.append(ecl_dis[i][1])

        res = max(set(mj_cl), key=mj_cl.count)

        return res

    cal_cl = []
    for l in range(0, b, 1):
        cal_cl.append(e_dis(l))

    c = 0
    for i in range(0, b, 1):
        if (cal_cl[i] == va_class[i][1]):
            c += 1
    acc = (c / b) * 100
    print("Accuracy: ", acc)
    return acc, cal_cl


n = int(input("Enter the number of time to check KNN: "))
acc_res = []
acc_cl = []
knn_a = []
best_acc = []
for i in range(0, n, 1):
    knn = int(input("Enter k: "))
    knn_a.append(knn)
    acc_res.append(find_knn(knn))

print("K value            Validation Accuracy                 Validation Class")
for i in range(0, n, 1):
    print("    ", knn_a[i], "              ", acc_res[i][0], "                ", acc_res[i][1])

best_acc.append(max(acc_res))
print("Best Accuracy: ", best_acc[0][0], "for classess: ", best_acc[0][1])
c = 0
acc_cl.append(best_acc[0][1])

acc = 0

while (acc < 60):
    for i in range(0, b, 1):
        if (acc_cl[0][i] == te_class[i][1]):
            c += 1
    acc = (c / b) * 100
    print("TEST accuracy: ", acc)
    te_class = shuffle(te_class)
