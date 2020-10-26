<!--
 * @Author: chenzhen
 * @Date: 2019-07-09 11:36:06
 * @LastEditTime: 2020-10-26 16:10:10
 * @LastEditors: chenzhen
 * @Description:
-->
# MatrixSlow - Deep Learning Framework in Python

[中文版](README.md)

![avatar](book.png)

### Introduction
In 1984, Professor Andrew S. Tanenbaum, author of Modern Operating Systems (commonly known as the Circus Book), developed Minix, an operating system for teaching purposes.Inspired by it (how much is unknown), Linus Torvalds created the Linux system. Here we decided to follow the example of our predecessors and implement a computational graph-based machine learning/deep learning framework from scratch in Python, which we call MatrixSlow.

The original purpose of implementing MatrixSlow is to share learning within our team, and to deepen the "understanding" of the theory by combining machine learning theory, principles and engineering practice in a hands-on "build" process. However, the source code of modern deep learning frameworks, such as TensorFlow, is already too complex for learning, so we decided to implement an extremely simple deep learning framework: MatrixSlow (2K lines of core code, relying only on Numpy). This framework is based on computational graphs, supports auto-differentiation and gradient optimization, and implements some common algorithms (LR, FM, DNN, CNN, RNN, W&D, etc.) and some engineering techniques (trainers, distributed training, services) on top of it. The humble name is to indicate that it is only a framework for teaching, supports only 2D matrices, does not consider computational optimization and therefore runs somewhat sloooowly.

In addition to open-sourcing the MatrixSlow code, we also put together some of the thoughts and implementation details of MatrixSlow's design into a book called "Deep Learning Framework in Python", which was published by People's Daily Publishing House (Turing Original) and is available for purchase on all major e-commerce platforms.



- [JD](https://item.jd.com/12994556.html)
- [Dangdang](http://product.dangdang.com/29139156.html)
- [Taobao](https://detail.tmall.com/item.htm?spm=a230r.1.14.110.52abd576UEklUs&id=628890432853&ns=1&abbucket=2)

We have written as detailed as possible in the code comments, even if you do not buy the book, through the reading of the source code can understand this "small but complete" framework, and from which to learn and understand the principles behind machine learning.

The famous physicist, Nobel Prize winner [Richard Feynman](https://en.wikipedia.org/wiki/Richard_Feynman) wrote on the blackboard in his office: "What I cannot create, I do not understand. "MatrixSlow and this book are a small exercise in the ideas of the great philosophers of science!


### Characteristics

- Based on computational graphs that can be used to build most machine learning models
- Support for automatic differential derivation
- Support for stochastic gradient descent optimization algorithms and several important variants thereof
- Support for common model evaluation algorithms
- Support for model saving and loading
- Supports PS and Ring AllReduce distributed training
- Support model serving

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
- matrixslow: The core code, including implementation of computational graphs, auto-differentiation and optimization, model saving and loading, and distributed training.
- matrixslow_serving: generic model inference service, similar to tensorflow serving
- example: follow the chapters in the book for examples, such as ch02 which shows how to implement a simple class LR model with matrixslow, and ch11 which demonstrates how to conduct distributed training.
