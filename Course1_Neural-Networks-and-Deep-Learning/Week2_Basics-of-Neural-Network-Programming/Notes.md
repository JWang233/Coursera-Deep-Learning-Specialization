
# 1. Logistic Regression as a Neural Network

## 1.1 Binary Classification

**Notations:**

- Dataset:  $(x,y), x\in \mathbb{R}^{n_{x}}, y \in \{0, 1\}$, where $n_{x}$ is the number of pixels in all bands

- Training and Testing data: $ m_{train/test} = \{(x^{1}, y^{1}), ..., (x^{m}, y^{m}) \} $

- Input data for neural network: 
    - $$X = [x^{1}, x^{2}, ..., x^{m}] $$
    where $X \in \mathbb{R}^{n_{x} \times m}$
    
    - $$Y = [y^{1}, y^{2}, ..., y^{m}] $$
    where $X \in \mathbb{R}^{1 \times m}$

## 1.2 Logistic Regression

**Basic Concept:**


- Given $X$ wants $\hat{y} = P(y=1 | x)$

    where $X \in \mathbb{R}^{n_{x}}$, $0 \leq \hat{y} \leq 1$
- Parameters: $\underline{\omega} \in \mathbb{R}^{n_{x}}$, $b \in \mathbb{R}$


- Outputs: 

    $ \hat{y} = \sigma \: ( \omega ^{T}x + b) $

    where $ \sigma(z) = \frac{1}{1 + e^{-z}} $
    
    <img src = 'imgs\sigmoid.png'>
    

## 1.3 Logistic Regression Cost Function

**Loss (Error) Function:**

$$\mathcal{L} (\hat{y}, y) = -( y \;\log \hat{y} + ( 1 - y) \; \log( 1 - \hat{y}) ) $$

Why this? 
- If $ y = 1: \: p(y|x) = \hat{y}$. The chance y given x is equal to 1.
- If $ y = 0: \: p(y|x) = 1 - \hat{y} $. The chance y given x is equal to 0.
- Thus, the probability can be expressed as $$ p(y|x) = \hat{y}^{y} \; (1 - \hat{y})^{1 - y} $$

- Since the $ \log()$ function is **strictly monotonically increasing, maximizing $p(y|x)$ is equal to maximize $\log(p(y|x) $. Thus, $$ \log(p(y|x) = y \log \hat{y} + ( 1 - y)  \log( 1 - \hat{y}) = - \mathcal{L}(\hat{y}, y) $$
    
    where, maximizing the probability is the same as minimizing the loss function.

**Cost Function:**

$$J(w,b) = \frac{1}{m} \; \sum _{i=1}^{m} \mathcal{L} (\hat{y} \; ^{(i)}, y^{(i)}) = - \frac{1}{m} \; \sum _{i=1}^{m}[ y^{(i)} \log \hat{y} \; ^{(i)} + ( 1 - y^{(i)} )  log( 1 - \hat{y} \; ^{(i)})) ]  $$

Why this? 
- Assume data is i.i.d
- Maximizing $$ \log P(labels \; in \; training \; set) = \log \prod_{i = 1}^{m} p(y^{(i)}, x^{(i)}) $$ $$ \log P(...) = \sum_{i=1}^{m} p(y^{(i)}, x^{(i)}) = -\sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) $$ 

- Using the principle, Maximum Likelihood Estimation, to maximize the probability, so the cost function is minimized. 


**Difference between Loss & Cost Function:**
- The loss function computes the error for a single training example
- The cost function is the average of the loss functions of the entire training set


## 1.4 Gradient Descent

**Learn\Train the Parameters**

- Goal: want to find $w,b$ that minimize $J(w,b)$, which is defined as a convex function for simplfying the optimization

<img src = 'imgs\gd-ill.jpg'>

- Algorithm:

    Repeat:
$$w \: := \: w - \alpha \; \frac{\partial J(w,b)}{\partial w} $$
$$b \: := \: b - \alpha \; \frac{\partial J(w,b)}{\partial b} $$
    
    where $\alpha$ is the learning rate controlling how big a step we take on each iteration, $\frac{\partial J(w)}{\partial w}$, $dw$ used in the following section, is the partial derivative on $w$ representing the basic update or the change we want to make to the parameter $w$.


## 1.5 Computation Graph

**Computational Graph:**

- forward propogation/pass: 

    <img src = 'imgs\computational-graph.jpg'>
    
- Final Output Variable: $J$
    - Notation: $$dvar := \frac{d \;FinalOutputVar}{d \; var}$$
    
    the derivative of a final output variable with respect to various intermediate quantities.

## 1.6 Derivatives with a Computation Graph

**Chain Rule:**

- Using the same graph in the above image

- Backward calculation: 
    - Follow the direction from the right to the left to compute the derivatives

$$ \frac{dJ}{dv} = 3; \:\:\:\:\: \frac{dJ}{da} = \frac{dJ}{dv} \frac{dv}{da} = 3 \times 1 = 3 $$

## 1.7 Logistic Regression Gradient Descent

**Problem recap:**
<img src = 'imgs\logis-gd-problem.jpg'>
   
**Goal:**

- Adjust $w_{1}, w_{2}, b$ to minimize the loss function $J$
    
    <img src = 'imgs\logis-gd-goal.jpg'>
    
**Backward derivatives:**

- Derivative of the logistic loss function: 

$$ da = \frac{d \; \mathcal{L}(a,y)}{da} = -\frac{y}{a} + \frac{1-y}{1-a} $$

- Derivative of the Sigmoid function: 

$$ \frac{d}{dx} S(x) = \frac{d}{dx} \; \frac{1}{1 + e^{-x}} $$

$$ = \frac{e^{-x}}{(1 + e^{-x})^{2}} $$ 

$$ = \frac{-1 + 1 + e^{-x}}{(1+ e^{-x})^{2}} $$ 

$$ = \frac{1}{1+ e^{-x}} - \frac{1}{(1+ e^{-x})^{2}} $$ 

$$ = \frac{1}{1+ e^{-x}}(1 - \frac{1}{1+ e^{-x}}) $$ 

$$ = S(x)\;(1 - S(x)) $$


- 
    - The reason why the derative is expressed in ths way is that we can use what we already calculated in the forward computation into the backward computation for saving more time.
    
- Thus, $$ dz = da \; \frac{da}{dz} = (-\frac{y}{a} + \frac{1-y}{1-a}) \; a \; (1-a) = a - y $$ 

$$ dw_{1} = x_{1} \: dz$$ $$ dw_{2} = x_{2} \: dz$$ $$ db = dz$$ 

- Algorithm: repeat $$ w_{1} := w_{1} - \alpha \: dw_{1} $$ $$ w_{2} := w_{2} - \alpha \: dw_{2} $$ $$ b := b - \alpha \: db $$

## 1.8 Gradient Descent on m Training Examples

**Overall Algorithms:**

- Initilization: $ J = 0, dw_{1} = 0, dw_{2} = 0, db = 0$ 

- For i = 1 to m 
    $$ z^{i} = w^{T}x^{i} + b $$ 
    $$ a^{(i)} = \sigma \; (z^{i}) $$
    
    $$ J += -(y^{i}\log a^{i} + (1 - y^{i})\log (1 - a^{i}))  $$
    $$ dz^{i} = a^{i} - y^{i} $$
    $$ dw_{1} \: += x^{i}_{1} \: dz^{i} $$
    $$ dw_{2} \: += x^{i}_{2} \: dz^{i} $$
    $$ db \: +=  dz^{i} $$
    
- Mean: $ J \; /= m, \: dw_{1} \; /= m, \: dw_{2} \; /= m, \: db \; /= m $
    
- Updates: $$ w_{1} := w_{1} - \alpha \: dw_{1} $$ $$ w_{2} := w_{2} - \alpha \: dw_{2} $$ $$ b := b - \alpha \: db $$

- Drawbacks: for loop is gonna slow down the entire process, especitally when there are a large number of parameters. Thus, the best solution is vectorization.

**Vectorized Algorithms:**

- Forward: 

$$ Z = w^{[T]} + b $$ 

$$ A = \sigma (Z) $$ 

- Backword: 

$$ dZ = A - Y $$ 

$$ dw = \frac{1}{m} \; X \;dZ^{T} $$ 

$$ db = \frac{1}{m} \; \text{np.sum($dZ$)} $$

$$ w := w - \alpha \; dw $$ 

$$ b := b - \alpha \; db $$

-------

# 2. Python and Vectorization

## 2.1 Vectorization

- Vectorization is much faster than the for loop! See the example below:


```python
import numpy as np
import time 

nNum = 1000000
a = np.random.rand(nNum)
b = np.random.rand(nNum)

tic = time.time()
c = np.dot(a, b) 
toc = time.time()
print(c)
print("Vectorized version:" + str(1000*(toc-tic)) + "ms")

c = 0
tic = time.time() 
for i in range(nNum): 
    c += a[i] * b[i]
toc = time.time()
print(c)
print("For loop:" + str(1000*(toc-tic)) + "ms")

```

    249834.3002996213
    Vectorized version:0.0ms
    249834.30029962066
    For loop:355.0863265991211ms
    

## 2.2 Neural Network Programming Guideline:

- Whenever possible, avoid explicit for-loops! 

## 2.3 Broadcasting in Python

**General Principle:**

- A matrix (m,n)  +-x/ another matrix (1,n) or (m,1), the second matrix will be copied to become the matrix with the same size

In python operation, aix = 0 means the operation in vertical line (column), while axis = 1 means the operation in horizontal line (row)

## 2.4 A Not on Python Vector


```python
# vector
a = np.random.randn(5, 1)
print(a)
print(np.shape(a))
print("This is a 5-by-1 vector. ALWAYS use this structure!\n")

# rank 1 array
a = np.random.randn(5)
print(a)
print(np.shape(a))
print("This is a rank 1 array. NEVER use this structure!\n")

# fix rank 1 array 
a = a.reshape(5,1)
print(np.shape(a))
print("The rank 1 array has been reshaped to a vector!")
```

    [[ 0.71984454]
     [ 0.29621368]
     [-0.25412647]
     [-1.26199709]
     [-0.03885003]]
    (5, 1)
    This is a 5-by-1 vector. ALWAYS use this structure!
    
    [ 0.21971449  1.45976932  1.75588662  0.84489119 -0.83085174]
    (5,)
    This is a rank 1 array. NEVER use this structure!
    
    (5, 1)
    The rank 1 array has been reshaped to a vector!
    

-----

# Quiz

**Matrix and Element-wise Multiplication:** 
- np.dot(a,b) performs a matrix multiplication on a and b

- a*b performs an element-wise multiplication.

-----------

# Assignments

## Notes: 

**1. Normalizing Rows:**

- After **normalization**, gradient descent converges faster so it often leads to a better performance!

- Method: dividing each row vector of x by its norm (square root of the sum of squares). Sometimes, normalizing imagse is by subtracting dataset's image mean instead of each image mean in DL. 

    - The reason we do both of those things is because in the process of training our network, we're going to be multiplying (weights) and adding to (biases) these initial inputs in order to cause activations that we then backpropogate with the gradients to train the model.
    
    We'd like in this process for each feature to have a similar range so that our gradients don't go out of control (and that we only need one global learning rate multiplier).

    Another way you can think about it is deep learning networks traditionally share many parameters - if you didn't scale your inputs in a way that resulted in similarly-ranged feature values (ie: over the whole dataset by subtracting mean) sharing wouldn't happen very easily because to one part of the image weight w is a lot and to another it's too small. 
    
    (From [stackoverflow](https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current))

- One example, if $$x = 
\begin{bmatrix}
    0 & 3 & 4 \\
    2 & 6 & 4 \\
\end{bmatrix}$$ then $$\| x\| = np.linalg.norm(x, axis = 1, keepdims = True) = \begin{bmatrix}
    5 \\
    \sqrt{56} \\
\end{bmatrix} $$and        $$ x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
    0 & \frac{3}{5} & \frac{4}{5} \\
    \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
\end{bmatrix}$$ Note that you can divide matrices of different sizes and it works fine: this is called broadcasting and you're going to learn about it in part 5.


- Image Dataset is more convenient and works almost as well to jsut scale the data to [0, 1]. 

**2. Softmax Function:**

- $ \text{for } x \in \mathbb{R}^{1\times n} \text{,     } softmax(x) = softmax(\begin{bmatrix}
    x_1  &&
    x_2 &&
    ...  &&
    x_n  
\end{bmatrix}) = \begin{bmatrix}
     \frac{e^{x_1}}{\sum_{j}e^{x_j}}  &&
    \frac{e^{x_2}}{\sum_{j}e^{x_j}}  &&
    ...  &&
    \frac{e^{x_n}}{\sum_{j}e^{x_j}} 
\end{bmatrix} $ 

- $\text{for a matrix } x \in \mathbb{R}^{m \times n} \text{,  $x_{ij}$ maps to the element in the $i^{th}$ row and $j^{th}$ column of $x$, thus we have: }$  $$softmax(x) = softmax\begin{bmatrix}
    x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
    x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{m1} & x_{m2} & x_{m3} & \dots  & x_{mn}
\end{bmatrix} = \begin{bmatrix}
    \frac{e^{x_{11}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{12}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{13}}}{\sum_{j}e^{x_{1j}}} & \dots  & \frac{e^{x_{1n}}}{\sum_{j}e^{x_{1j}}} \\
    \frac{e^{x_{21}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{22}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{23}}}{\sum_{j}e^{x_{2j}}} & \dots  & \frac{e^{x_{2n}}}{\sum_{j}e^{x_{2j}}} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \frac{e^{x_{m1}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m2}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m3}}}{\sum_{j}e^{x_{mj}}} & \dots  & \frac{e^{x_{mn}}}{\sum_{j}e^{x_{mj}}}
\end{bmatrix} = \begin{pmatrix}
    softmax\text{(first row of x)}  \\
    softmax\text{(second row of x)} \\
    ...  \\
    softmax\text{(last row of x)} \\
\end{pmatrix} $$

**3. A trick in np.reshape()**

- -1 is used to let the numpy to figure the dimension of the matrix/vector by itself. 

- A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use: 
```python
X_flatten = X.reshape(X.shape[0], -1).T 
```
    
- More details can be found in [stackoverflow](https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape)


**4. Learning Rate Tuning:**

- Different learning rates $\alpha$ give different costs and thus different predictions results. The learning rate determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

<img src = 'imgs\learning-rate-tuning.png'>

- If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost). 

- A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.

- In deep learning, we usually recommend that you: 
    
    - Choose the learning rate that better minimizes the cost function.
    
    - If your model overfits, use other techniques to reduce overfitting. (We'll talk about this in later videos.) 

## Projects: 

**Logistic Regression with a Neural Network mindset v5**

- Framework: 

    <img src = 'imgs\proj-cat.png'>
    
- Mathematical expression of the algorithm: 

    For one example $x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

    The cost is then computed by summing over all training examples:
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$


- Final Outputs: 
    
    - Cost: 
    ```Python
    Cost after iteration 0: 0.693147
    Cost after iteration 100: 0.584508
    Cost after iteration 200: 0.466949
    Cost after iteration 300: 0.376007
    Cost after iteration 400: 0.331463
    Cost after iteration 500: 0.303273
    Cost after iteration 600: 0.279880
    Cost after iteration 700: 0.260042
    Cost after iteration 800: 0.242941
    Cost after iteration 900: 0.228004
    Cost after iteration 1000: 0.214820
    Cost after iteration 1100: 0.203078
    Cost after iteration 1200: 0.192544
    Cost after iteration 1300: 0.183033
    Cost after iteration 1400: 0.174399
    Cost after iteration 1500: 0.166521
    Cost after iteration 1600: 0.159305
    Cost after iteration 1700: 0.152667
    Cost after iteration 1800: 0.146542
    Cost after iteration 1900: 0.140872
    Cost after iteration 2000: 0.135608
    Cost after iteration 2100: 0.130708
    Cost after iteration 2200: 0.126137
    Cost after iteration 2300: 0.121861
    Cost after iteration 2400: 0.117855
    Cost after iteration 2500: 0.114093
    Cost after iteration 2600: 0.110554
    Cost after iteration 2700: 0.107219
    Cost after iteration 2800: 0.104072
    Cost after iteration 2900: 0.101097
    Cost after iteration 3000: 0.098280
    Cost after iteration 3100: 0.095610
    Cost after iteration 3200: 0.093075
    Cost after iteration 3300: 0.090667
    Cost after iteration 3400: 0.088374
    Cost after iteration 3500: 0.086190
    Cost after iteration 3600: 0.084108
    Cost after iteration 3700: 0.082119
    Cost after iteration 3800: 0.080219
    Cost after iteration 3900: 0.078402
    ```
    
    - Accuracy: 
    ```Python
    train accuracy: 99.52153110047847 %
    test accuracy: 70.0 %
    ```
    Definitely overfitting.  
    
    - Learning Rate Selection: 
    
    <img src = 'imgs\learning-rate-tuning.png'>
