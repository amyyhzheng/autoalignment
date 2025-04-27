
The Richardson-Lucy (RL) algorithm is derived from Bayes' theorem, which provides a framework for updating probabilities based on new evidence. Here's how the derivation proceeds:

1. Bayes' theorem is applied to image restoration:

$$
P(f(x)|g(x)) = \frac{P(g(x)|f(x))P(f(x))}{P(g(x))}
$$

Where f(x) is the true image and g(x) is the observed blurred image[^2].

2. The likelihood P(g(x)|f(x)) is modeled using a Poisson distribution, which is appropriate for image noise:

$$
P(g(x)|f(x)) = \prod_{x} \frac{(f(x) \otimes h(x))^{g(x)} e^{-(f(x) \otimes h(x))}}{g(x)!}
$$

Where h(x) is the point spread function (PSF)[^2].

3. The negative log-likelihood is then minimized:

$$
L(f(x)) = \sum_{x} (f(x) \otimes h(x) - g(x) \ln(f(x) \otimes h(x)))
$$

4. Through a series of mathematical manipulations and approximations, including Taylor expansion and convolution properties, the update rule is derived:

$$
f^{n+1} = \left(\frac{g(x)}{f^n \otimes h(x)} \otimes h\right) f^n
$$

This iterative update rule forms the core of the Richardson-Lucy algorithm[^2].

By deriving the algorithm from Bayes' theorem, it inherently incorporates prior knowledge about the image formation process and noise characteristics, leading to a statistically motivated approach for image deblurring[^1][^2].
