# Neural Network basics for Multi-Layer Perceptrons (MLPs)

This document describes the basic idea of how a multi-layer perceptron predicts an output given an input. In this specific case, (the idea)


$$ \frac{\partial L}{\partial {}_lW} = \frac{\partial L}{\partial {}_lz}\cdot\frac{\partial {}_lz}{\partial {}_lW} \tag{Chain Rule}$$

$$ \frac{\partial L}{\partial {}_lW} = \frac{\partial L}{\partial {}_lz}\cdot\frac{\partial}{\partial {}_lW}\left({}_lW{}_{l-1}a + {}_lb\right)$$

$$ \frac{\partial L}{\partial {}_lW} = \frac{\partial L}{\partial {}_lz}\cdot {}_{l-1}a^T \tag{1}$$
\
where ${}_{l-1}a$ is the "activated" vector from the previous layer, $l - 1$.

Similarly for relative rate of change of $L$ w.r.t. the bias $b$,
$$ \frac{\partial L}{\partial {}_lb} = \frac{\partial L}{\partial {}_lz}\cdot\frac{\partial}{\partial {}_lb}\left( {}_lW{}_{l-1}a + {}_lb \right) $$

$$ \frac{\partial L}{\partial {}_lb} = \frac{\partial L}{\partial {}_lz}\cdot 1 \tag{2}$$


---

The loss function used in our case is the cross-entropy loss function. 
$$ L = -\sum_k y_k\ln(\hat y_k)$$

where $y$ is the correct answer (in this case, a one hot vector) and $\hat y$ is the output of the multi-layer perceptron.

$$ \frac{\partial L}{\partial\hat y_k} = -\frac{\partial}{\partial\hat y_k} \left( y_0\ln(\hat y_0) + y_1\ln(\hat y_1) + \cdots + y_k\ln(\hat y_k) + \cdots \right) $$

$$ \frac{\partial L}{\partial\hat y_k} = -\left( 0 + 0 + \cdots + \frac{\partial}{\partial\hat y_k}\ y_k\ln(\hat y_k) + \cdots \right) $$

$$\boxed{\ \ \frac{\partial L}{\partial\hat y_k} = -\frac{y_k}{\hat y_k}\ \ } \tag{3}$$


---

A softmax function is used at the final layer to activate the resultant vector $z$, converting it into a probability vector $\hat y$.

$$ \hat y_k = \frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{z_k}}{S} $$

Consider an arbitrary logit $z_i$. The relative rate of change of a specific probability $\hat y_k$ w.r.t. that specific logit in a (un-activated) vector $z$ is,

$$ \frac{\partial\hat y_k}{\partial z_i} = \frac{\partial}{\partial z_i}\left(\frac{e^{z_k}}{e^{z_i} + S'}\right) $$

where $S'$ is just the sum of all exponentials excluding the $z_i$ logit.

### when $i = k$,

$$ \frac{\partial\hat y_k}{\partial z_k} = \frac{\partial}{\partial z_k}\left( \frac{e^{z_k} + S' - S'}{e^{z_k} + S'} \right) = \frac{\partial}{\partial z_k}\left( 1 -  \frac{ S'}{e^{z_k} + S'} \right)$$

$$ \frac{\partial\hat y_k}{\partial z_k} = S'\frac{e^{z_k}}{\left(e^{z_k} + S'\right)^2} $$

$$ = S'\frac{e^{z_k}}{S^2} $$

$$ = \frac{S'}{S}\hat y_k $$

$$ = \frac{S - e^{z_k}}{S}\hat y_k $$

$$ = (1 - \hat y_k)\hat y_k $$

$$ \frac{\partial\hat y_k}{\partial z_k} = (1 - \hat y_i)\hat y_k \tag{4}$$

### when $i \ne k$,

$$ \frac{\partial\hat y_k}{\partial z_i} = -e^{z_k}\frac{e^{z_i}}{\left( e^{z_i} + S'\right)^2} $$

$$ = -\frac{e^{z_k}e^{z_i}}{S^2} $$

$$ = -\frac{e^{z_i}}{S}\hat y_k $$

$$ = -\hat y_i \hat y_k $$

$$ \frac{\partial\hat y_k}{\partial z_i} = (0 - \hat y_i)\hat y_k \tag{5} $$

\
We can combine $(4)$ and $(5)$ in a more concise form using the **Kronecker-Delta** function. 

$$ \boxed{ \frac{\partial\hat y_k}{\partial z_i} = (\delta_{ik} - \hat y_i)\hat y_k } \tag{6} $$


---

We can now obtain the relative rate of change of the Loss $L$ w.r.t. a logit $z_i$ using the chain rule,

$$ \frac{\partial L}{\partial z_i} = \sum_k\frac{\partial L}{\partial \hat y_k}\cdot \frac{\partial\hat y_k}{\partial z_i} $$

Combining $(3)$ and $(6)$ gives,

$$ \frac{\partial L}{\partial z_i} =\sum_k-\frac{y_k}{\hat y_k}\cdot(\delta_{ik} - \hat y_i)\hat y_k $$

$$ = \sum_k y_k(\hat y_i - \delta_{ik}) $$

$$ = \sum_k (y_k\hat y_i - y_i) $$

$$ = \hat y_i\left( \sum_k y_k \right) - y_i $$

Since $y$ is a probability distribution (in this case, a one-hot vector), the sum of all its components must be 1.

$$ \frac{\partial L}{\partial z_i} = \hat y_i - y_i $$

$$ \boxed{\frac{\partial L}{\partial z} = \nabla_zL = \hat y - y} \tag{7} $$

---

Equation $(7)$ only works for the last layer. If we refer to the first layer (input to first hidden) as layer 0 and let the number of layers be $N$,

$$ \Delta_{N-1} = \frac{\partial L}{\partial z} = \hat y - y $$

Then updating the weights and biases can concisely written as,

$$ W \leftarrow W - \eta\Delta_{N-1}\cdot a^T $$

$$ b \leftarrow b - \eta\Delta_{N-1} $$


> From here on, notation such as $z_0$ will refer to the resultant un-activated (logit) vector by transforming layer 0. Previously, this notation was used to describe a specific logit component in the vector.


$$ \frac{\partial L}{\partial z_{N-2}} = \frac{\partial L}{\partial z_{N-1}}\cdot\frac{\partial z_{N-1}}{\partial z_{N-2}} \tag{Chain Rule} $$

$$ \Delta_{N-2} = \Delta_{N-1}\cdot\frac{\partial z_{N-1}}{\partial z_{N-2}} $$

$$ = \Delta_{N-1}\cdot\frac{\partial}{\partial z_{N-2}}\left(W_{N-1}\cdot a_{N-2} + b_{N-1}\right) $$

$$ = \Delta_{N-1}\cdot\frac{\partial}{\partial z_{N-2}}\left(W_{N-1}\cdot {\rm relu}(z_{N-2}) + b_{N-1}\right)$$

$$ = \Delta_{N-1}\cdot W_{N-1}\cdot{\rm H}(z_{N-2}) $$

\
where ${\rm H}(x)$ is the unit heaviside function.



