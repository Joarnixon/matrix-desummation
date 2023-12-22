
# Simple introduction can be found at demo.ipynb


# Abstract:

### This project was a small experiment.
### Which after some research came up to be a view from another perspective on SVD, NMF and Regression.
### Sometimes inspiration for a problem can come from looking at it from another perspective or even from a completely different field of research, so I would recommend you to take a glance at what I did.

# What I mean by desummation?

By that, I mean representing a given matrix A as a sum of matrices $\cdot B_i$ with corresponding weights or coefficients $\cdot w_i$.

$$
  A = w_1 \cdot B_1 + w_2 \cdot B_2 + \ldots + w_n \cdot B_n
$$

I assume that the reader is familiar with the basics of linear algebra and can easily recall the facts used.

### Here are some popular mathematical theorems on this topic:

 - Spectral decomposition of a symmetric matrix:

$$
  A = Q \Lambda Q^T = \sum_{i=1}^{n} \lambda_i*u_iu_i^T = \sum_{i=1}^{n} \lambda_i * U_i
$$

#### Actually this theorem works for any matrix, but with some modification:
- Spectral decomposition for diagonalizable matrix:

$$
  A = S \Lambda S^{-1} = \sum_{i=1}^{n} S  \begin{pmatrix}
    0 & 0 & 0 & \ldots & 0 \\
    0 & \ddots & 0 & \ldots & 0 \\
    0 & 0 & \lambda_i & \ldots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & \ldots & 0 \\
  \end{pmatrix}S^{-1}= \sum_{i=1}^{n}M_i
$$
- We can even construct basis with $n^2$ dimension:

$$
  B_{ij} = \begin{pmatrix}
    0 & 0 & 0 & \ldots & 0 \\
    0 & \ddots & 1 & \ldots & 0 \\
    0 & 0 & 0 & \ldots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & \ldots & 0 \\
  \end{pmatrix} i,j=1 \ldots n
$$

#### Alternatively, one may use Singular Value Decomposition with the same approach.

###### I randomly encountered a similar theorem for tensors, but for now, I will stick with a simpler perspective.

### One thing that unites these methods is that the matrices have a lower rank than the original.
 - This is one of the methods for reducing the dimensionality of feature space in ML, but we don't actually decompose it into a sum.
 
 - Instead we take smallest singular values and replace with zero obtaining lower rank approximation that (can be proven) will be the best amongst all other matrices that rank (by Frobenius norm).

 - Or find sort of projection on a matrix space with lower rank using Non-Negative Matrix Factorization.

### Motivation:
- Very naive and simple: experiment with matrix topology and make some research.
- Perhaps certain matrices in the given problem `can approximate`
other matrices `better` than the others?
- What if we aply this to `pictures`? _We might lose some information in process, but instead we gain $\ldots$ weights(*)_
- What if these weights(*) represent some sort of `coordinate space` (with a fixed basis of $B_i$ matrices)

### All math and code can be found in `HowAndWhyItWorks.ipynb`. Open it in github or clone the repository
# Thank you for visiting
