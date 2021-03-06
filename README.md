# KNN-implementation


Dataset preparation (Randomly Split the dataset into Training (70%), Validation (15%) and Test (15%) set):
Train_set=[], Val_set=[], Test_set=[]
Shuffle your dataset list 
1.for each sample S in the dataset:
2.	generate a random number R in the range of [0,1]
3.	if R>=0 and R<=0.7:
4.		append S in Train_set 
5.	elif R>0.7 and R<=0.85:
6.		append S in Val_set
7.	else:
8.		append S in Test_set


KNN Classification (Use Iris data):
K = 5
1.for each sample V in the VALIDATION set:
2.	for each sample T in the TRAINING set:
3.		Find Euclidean distance between Vx (features->N-1) and Tx (features->N-1)
4.		Store T and the distance in list L
5.	Sort L in ascending order
6.	Take the first K samples
7.	Take the majority class from the K samples (this is the detected class for sample V)
8.	Now, check if this class is correct or not
9.Calculate validation accuracy = (correct VALIDATION samples)/(total VALIDATION samples) * 100

Calculate validation accuracy in a similar way for K = 1, 3, 5, 10, 15, ...
Make a table with 2 columns: K and Val_acc (doc file)
Now, take the K with highest Val_acc
Use this best K to determine Test_acc (Simply replace VALIDATION set of line 1. with TEST set)


KNN Regression (Use diabetes data):
K = 5, Error = 0
1.for each sample V in the VALIDATION set:
2.	for each sample T in the TRAINING set:
3.		Find Euclidean distance between Vx and Tx
4.		Store Tx and the distance in list L
5.	Sort L in ascending order
6.	Take the first K samples
7.	Take the average output of the K samples (this is the determined output for sample V)
8.	Error = Error + (V true output - V determined output)^2
9.Calculate Mean_Squared_Error = Error/(total number of samples in VALIDATION set)

Calculate Mean_Squared_Error in a similar way for K = 1, 3, 5, 10, 15, ...
Make a table with 2 columns: K and Mean_Squared_Error (doc file) 
Now, take the K with minimum Mean_Squared_Error
Use this best K to determine Mean_Squared_Error for Test set (Simply replace VALIDATION set of line 1. with TEST set)
