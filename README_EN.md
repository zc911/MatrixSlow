<!--
 * @Author: chenzhen
 * @Date: 2019-07-09 11:36:06
 * @LastEditTime: 2020-10-29 14:38:48
 * @LastEditors: chenzhen
 * @Description:
-->
# MatrixSlow - Deep Learning Framework in Python

[中文版](README.md)

![avatar](book.png)

### Introduction
In 1984, Professor Andrew S. Tanenbaum, author of Modern Operating Systems (commonly known as the Circus Book), developed a teaching operating system [Minix](https://www.minix3.org). Inspired by it (how much is unknown), Linus Torvalds created the Linux system. Here we decided to follow the example of our predecessors and implement from scratch a computational graph-based machine learning/deep learning framework in Python, which we call MatrixSlow.

The original intent of implementing MatrixSlow was to share learning within our team. By combining machine learning theory, principles, and engineering practice, we can deepen our "understanding" of the theory through hands-on "building". The deep learning framework is a perfect example of this. However, the source code of modern deep learning frameworks such as TensorFlow is already too complex for learning, so we decided to implement a simple deep learning framework MatrixSlow in pure Python (about 2K lines of core code, depending only on Numpy). This framework is based on computational graph and supports automatic derivation and gradient descent optimization algorithms (and variants). (We have built a number of classical models using this framework, including LR, FM, DNN, CNN, RNN, W&D, DeepFM, etc.). The framework also includes a number of engineering solutions, including trainers, distributed training, model deployment, and services. The modest name MatrixSlow was taken to indicate that it is only a framework for teaching purposes, supports only second-order tensor (Matrix, but no loss in expressiveness, see Chapter 12, Section 1 of this book for details), does not take computational optimization into account, and therefore runs somewhat sloooowly.

In addition to the open source code of MatrixSlow, we also put together some of the thoughts and implementation details when designing MatrixSlow into a book "Deep Learning Framework in Python", which was published by People's Daily Publishing House (Turing Original) and is available for purchase on all major e-commerce platforms.


- [JD](https://item.jd.com/12994556.html?cu=true&utm_source=zhuanlan.zhihu.com&utm_medium=tuiguang&utm_campaign=t_1001542270_1002093764_0_1956949436&utm_term=71be62a1a29845aaa4ac74b359c06e49)
- [Dangdang](http://product.dangdang.com/29139156.html)
- [Taobao](https://detail.tmall.com/item.htm?spm=a230r.1.14.110.52abd576UEklUs&id=628890432853&ns=1&abbucket=2)

We have written as detailed as possible in the code comments, even if you do not buy the book, through the reading of the source code can understand this "small but complete" framework, and from which to learn and understand the principles behind machine learning.

The famous physicist, Nobel Prize winner [Richard Feynman](https://en.wikipedia.org/wiki/Richard_Feynman) wrote on the blackboard in his office: "What I cannot create, I do not understand. "MatrixSlow and this book are a small exercise in the ideas of a great scientific philosopher.


### Characteristics

- Based on computational graphs, can be used to build most machine learning models.
- Support for automatic derivation.
- Supports stochastic gradient descent optimization algorithm and several of its important variants (e.g., RMSProp, ADAGrad, ADAM, etc.).
- Support for common model evaluation algorithms.
- Support for model saving and loading.
- Support PS and Ring AllReduce distributed training.
- Support for model serving.

### Dependencies
Core：
```
- python 3.7 or above
- numpy
```
Distributed training：
```
- protobuf
- grpc
```
Example：
```
- pandas
```

### Codes
```
.
├── README.md
├── benchmark
├── example
├── matrixslow
└── matrixslow_serving
```
- matrixslow: The core code section, including computational graphs, auto-derivation and optimization algorithms, model saving loading, and distributed training. matrixslow_serving: A generic model inference service, similar to tensorflow serving.
- matrixslow_serving: Generic model inference service, similar to tensorflow serving.
- example: ch05 introduces how to build and train multi-layer fully connected neural networks with matrixslow, and ch11 demonstrates how to run distributed training.
