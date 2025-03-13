
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
    dtype= tf.int64
)
print(random_num_genrator)
