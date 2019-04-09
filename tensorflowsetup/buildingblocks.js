// const tf = require('@tensorflow/tfjs-node');
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

// Tensors
const t1 = tf.tensor([1, 2, 3, 4, 2, 4, 6, 8], [2, 4]);
t1.print()
// Result
// [[1,2,3,4],
// [2,4,6,8]]
// Tensor without defining shape
const t2 = tf.tensor([1, 2, 3, 4], [2, 4, 6, 8]);
t2.print()
// tf.scalar()
// tf.tensor1d()
// tf.tensor2d()
// tf.tensor3d()

// Creating a tensore with all values 0 or 1
const t_zeros = tf.zeros([2, 3]);
t_zeros.print()
// Result
// [[0,0,0],
// [0,0,0]]

// Operations
const t3 = tf.tensor2d([1, 2], [3, 4])
t3.print()
const t3_squared = t3.square()
t3.print()

// Result
// [[1,4], [9,16]]


