import tensorflow as tf
import numpy as np

# note u can use tf.Variable or tf.Constand based on method

# creating a 1d array using tesorfloww:
tensor_1d = tf.Variable([4])
print(tensor_1d)

#creating a 2d array:
tensor_2d = tf.Variable([[34,56], [54,8]])
print(tensor_2d)

# creating a 3d array:
tensor_3d = tf.Variable([
    [[3,4],
     [4,6]],
    [[32,56],
     [67,88]]
])
print(tensor_3d)

#converting datatype:
tensor_1d = tf.Variable([3,4,5,7,8], dtype= tf.int16)
tensor_1d_cast = tf.cast(tensor_1d, dtype = tf.float32)
print(tensor_1d_cast)
#tf.cast is used to convert dtype of tensors

# creating ones array we will use a method tf.ones:
only_ones = tf.once([2,3], dtype= tf.int64)
onl_ones = tf.once([3,5],dtype= tf.bool)
print(only_ones)
print(onl_ones)

# creating zero array we will use a method tf.zeros:
only_zero = tf.zeros([3,4])
print(only_zero)

#generating number with tensor:
random_num_genrator = tf.random.uniform(
    [2,3],
    minval = 0,
    maxval = 100,
    dtype= tf.int64,
    seed = None,
    name = None
)
print(random_num_genrator)
#seed - use seed if you want to same value in other random number generator:
tf.random.set_seed(5)
print(tf.random.uniform([5,],maxval = 5,seed=10))
print(tf.random.uniform([5,],maxval = 5,seed=10))
print(tf.random.uniform([5,],maxval = 5,seed=10))
# the result of this value is same in every program if set_seed , shape and seed will be same.

#lets say i have created another random seed with same value: 
tf.random.set_seed(5)
print(tf.random.uniform([5,],maxval = 5,seed=10))
print(tf.random.uniform([5,],maxval = 5,seed=10))
print(tf.random.uniform([5,],maxval = 5,seed=10))
# result will be same as the upper one bcz seed , shape, set_seed is same.
#if we try to change anything result will be different.

# 2nd method:
norma_random = tf.random.normal(
    [2,3],
    mean = 10,
    stddv = 1
)
print(norma_random)# in this mean means it generate random number closesr to 10
#stddv means how far can go to 10
#genreate normal if you want number to be closer of your desire 
#genreate with uniform if you want just random number

# slicing elements:
varr = tf.variable([4,5,76,8,6,8,90])
print(varr)
print(varr[:])
varrr = tf.Variable([
    [[3,4,6],
     [46,4,3]],
    [[32,6,8],
     [2,6,7]]
])
print(varrr[0:,:,0])

#tensor math:
x_abs = ([3, -7.6, 34.5, -6])
tf.abs(x_abs)# it will print : 3. , 7.6 , 34.5, 6.0
#so abs(absolute) does change negative to positive ,+ve >+ve

#all math method do with their corresponding indexes..
# add:
t1_tensor = tf.Variable([3,5,6,8])
t2_tensor = tf.Variable([6,9,12,17])
print(tf.add(t1_tensor, t2_tensor))#output : 9, 14,18, 25
#reduce sum method:
red_sum = tf.Variable([
    [3,5,7],
    [1,-4,3],
    [-5,-3, 6]
])
print(tf.math.reduce_sum(red_sum)) #it will add all the number and give result: 13

#subtract:
print(tf.subtract(t1_tensor, t2_tensor))#output: -3, -4, -6, -9 

# multiply:
print(tf.multiply(t1_tensor, t2_tensor))

#divide:
print(tf.divide(t1_tensor, t2_tensor))

#what if there is an situation where 3/0 so it will infinite so for that case we use:
print(tf.math.divide_no_nan(t1_tensor, t2_tensor))
#it also divide but when there x/y where y = 0 then ans will be o for that particular index

#finding max,min value (in 1d):
arg_var = tf.Variable([4,67,3,56,7,43], dtype = tf.int32)
print(tf.math.argmax(arg_var))  #it will print -  1(index)
print(tf.math.argmin(arg_var))  #it will print -  2(index)
# inding max value (mutlidimensional):

arg_multi_var = ([
    [23,57,5,7],
    [46,85,89,4]
])
print(tf.math.argmax(arg_multi_var, 0)) # 0 means we willfind max value of each column
print(tf.math.argmax(arg_multi_var, 1)) # 1 means we well find max value of each row
print(tf.math.argmin(arg_multi_var, 0)) # 0 means we willfind minimun value of each column
print(tf.math.argmin(arg_multi_var, 1)) # 1 means we willfind min value of each row

#power method:
power = tf.Variable([[2,6], [1,4]], dtype = tf.int64)
power1 = tf.Variable([[1,4], [2,3]])
print(tf.pow(power, power1))#it will print : 2,1296 2, 64
#it multiplies by 1st row 1colum * 2nd row 1st column and as it is..

#reduce methods:
red = tf.Variable([
    [3,5,6,7],
    [-3,56,6,76],
    [4,7,5,6]
])
print(tf.math.reduce_sum(red)) #output: 178
#if we specify axis it will on the basis of axis:
print(tf.math.reduce_sum(red, axis= 0))#column #output : 4, 68, 17, 88
print(tf.math.reduce_sum(red, axis= 1)) #row #output: 21, ,22
#min:
print(tf.math.reduce_min(red)) #output: -3
#max:
print(tf.math.reduce_max(red)) #output: 76

print(tf.math.reduce_mean(red)) #output: mean all and give result

#sigmoid: if a +ve no is large then it will approach to 1
         # of a -ve no is large then it will approach to 0
#   for +ve:
x = tf.math.sigmoid([2.0,667.0,100.0,345.0])
print(tf.math.sigmoid(x))
# for -ve:
y = tf.math.sigmoid([-5.0,-100.0,-576.0])
print(tf.math.sigmoid(y))

# top_k : return highest value and also index of each array(k =1 default)
top_kk = tf.Variable([
    [43,4,24,76],
    [56,67,89,24]
])
print(tf.math.top_k(top_kk)) # return:  76, 89 and also: 3, 2(index)
#we can also define how many value we want by :
print(tf.math.top_k(top_kk, k=2))

#now some linear algebra:
# linalg.matmul:  this is for mutipling 2 matrices:
mat1 = tf.Variable([
    [2,45,6],
    [3,5,6]
])     
mat2= tf.variable([
    [2,4,55,7],
    [3,8,5,3],
    [3,5,8,9]  
])
print(tf.linalg.matmul(mat1, mat2))#we can also write this as:
print(mat1@mat2) #it will give same result
#for transpose of matrix :
print(tf.transpose(mat1)) #it will give transopose..

#cross product:
cross1 = tf.Variable([
    [1,3,4],
    [4,6,8]
])
cross2 = tf.Variable([
    [2,8,9],
    [7,4,8]
])
print(tf.linalg.cross(cross1, cross2))
     
#determinant: #should be square matrix and in float:
determinant = tf.Variable([
    [3,4,6],
    [4,7,8],
    [2,5,7] 
]) 
deter_cast = tf.cast(determinant, dtype= tf.float32) #we willchnage to float from int
print(tf.linalg.det(deter_cast)) 
# another method: without casting:
determin = tf.Variable([
    [2.0,3.0,5.0],
    [3,5,8]
    [2,5,9]
])
print(tf.linalg.det(determin)) #also gives result without casting 
 
# inverse method:
print(tf.linalg.inv(determin))
# transpose:
print(tf.linalg.matrix_transpose(determin))
# adjoint;
print(tf.linalg.diag(determin)) 
#exponential:
print(tf.linalg.expm(determin))

#multiplyig using numpy:
ra_num = np.array([ 
    [2,3,5,7],
    [3,5,7,6], 
    [3,1,7,8]
])
ra_num1 = np.array([
    [2,9,3],
    [4,7,8],
    [3,6,7],
    [3,6,8]
])
# mat__mul = np.matmul(ra_num, ra_num1)
#this will use to multiply 2 matrix
# print(mat__mul)

#to expand dimension of matrix:
expd = tf.Variable([[[2,4,6,6]]]) #3dimesion
exp_dim = tf.expand_dim(expd, axis=0)
print(exp_dim)  #now 4 dimension
#if we want to decrcease 1 dimension by 1 we will write:
squee = tf.squeeze(expd, axis=0)
print(squee) #now it has 2 dimension

exxpl = tf.Variable([[[2,3,5]]])
# print(exxpl.shape)
# exp = tf.expand_dims(exxpl, axis =0)
# print(exp)
# sque = tf.squeeze(exxpl, axis=0)
# print(sque)
for i in  range(3):
  sque = tf.squeeze(exxpl, axis=0)
  print(sque) #used to remove dimension of tensor
  
#   paddingmethod:
# if we want to make a matrix with some values and other value should we decide we can make by:
const1 = tf.variable([
    [2,3,5],
    [76,8,9]#so this is the value we want to create 
])
paddin = [[[2,3], [1,2]]]
print(tf.pad(const1, paddin, connstant_values = 1)) 
#here by default constant value is set by 0.
#that means we want to create a matrix where const1 values should be in direction
#in these directions 2,2 means we want to write matrix by lefting 2 rows from upper and 3 means we want 
#to write matrix by lefting 3rows form lower 
#and 1,2 menas we want to write our matrix lefting 1 from left side and 2 means we want to write lefting 2 columns right