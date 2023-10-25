# What is desummation?

Well, by that I mean representing a given matrix A as a sum of matrices $\cdot B_i$ with corresponding weights or coefficients $\cdot w_i$

$A = w_1 \cdot B_1 + w_2 \cdot B_2 + \ldots + w_n \cdot B_n$

I suppose that the reader is familiar with basics of linear algebra and can remember used facts with no difficulties.

### Some popular mathematical theorems on this topic:

 - Spectral decomposition of a symmetric matrix:
  $A = Q \Lambda Q^T = \sum_{i=1}^{n} \lambda_i*u_iu_i^T = \sum_{i=1}^{n} \lambda_i * U_i $

#### Actually this theorem works for any matrix but with some modification:
- Spectral decomposition for diagonalizable matrix:
  $$ A = S \Lambda S^{-1} = \sum_{i=1}^{n} S  \begin{pmatrix}
    0 & 0 & 0 & \ldots & 0 \\\
    0 & \ddots & 0 & \ldots & 0 \\\
    0 & 0 & \lambda_i & \ldots & 0 \\\
    \vdots & \vdots & \vdots & \ddots & \vdots \\\
    0 & 0 & 0 & \ldots & 0 \\\
\end{pmatrix}S^{-1} = \sum_{i=1}^{n}M_i $$

#### Or one may use Singular Value Decomposition with the same approach

##### I randomly encountered kinda same theorem for tensors, but for now I will stick with simpler perspective.

### One thing, that unites these methods is that the matrices are lower in rank than original
 - This is one of the methods of feature space dimensionality reduction in ML, but we actually don't decompose it to sum. 
 
 - Instead we take smallest singular values and replace with zero obtaining lower rank approximation that (can be proven) will be the best amongst all other matrices that rank (by Frobenius norm).

### Motivation:
- Very naive and simple: experiment with matrix topology and make some research.
- Maybe some matrices in given problem `can approximate` any matrices `better` ` than the others?
- What if we use this for `pictures`? We might lose some information in process, but instead we gain $\ldots$ weights(*)
- What if these weights(*) are some sort of `coordinate space` (of course with fixed basis of $B_i$ matrices)

# Now we are ready to start

- ### Open presentation.ipynb or clone the repository


### Links:


Spectral decomposition of symmetric matrix
- https://inst.eecs.berkeley.edu/~ee127/sp21/livebook/thm_sed.html

Diagonal desomposition
- https://mathworld.wolfram.com/MatrixDiagonalization.html