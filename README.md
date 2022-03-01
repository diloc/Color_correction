# ColorBayes Bayesian color constancy


Our goal is to correct the local color distortion on plant phenotyping images caused by non-uniform illumination. The corrected image will show the colors of individual plants as if they were taken under the same standard illuminant (D65). This color constancy approach has two main steps. The first step is to estimate an unknown illuminant's color and spatial distribution that causes the local color distortion. For this step, it is required a training dataset (ground truth), observed image data. Also, it is used the Bayes' rule and the maximum a posteriori (MAP). The second step is to transform the observed image using the chromaticity adaptation method.


<figure>
  <img src="https://github.com/diloc/Color_correction/blob/main/images/Figure_2_ColorLight_distribution4.png">
  <figcaption>
  Figure 1 Light environment characteristics in the HTPP system. (a) A top-view image capturing one hundred and forty-eight pots and twelve Macbeth ColorCheckers, illustrating non-uniform illumination. (b) Spatial distribution of the illumination color on the plant phenotyping scene. (c) Illumination color distribution on the Chromaticity diagram using the 1931 CIE colorimetric system coordinates. (d) Spectral Irradiance at (x = 20 cm and y = 20 cm) and (x = 0 cm and y = 6 cm), the wavelength range is from 325 nm to 785 nm on the horizontal axis and the spectral irradiance [μW cm-2 nm-1] on the vertical axis. Two peaks represent the primary emission LED lights at blue (450 nm) and red (630 nm) and a curve plateau between these peaks (550 - 600 nm).
  </figcaption>
</figure>

## Description

Our methods relies on the following assumptions:

- An observed image (5105 x 3075 pixels) is made up of three independent color channels <(k = {R,G,B})> and divided into areas that correspond to individual pot areas.
- A pixel class is assigned individually to segmented objects in a pot area such as plant and soil pixels. It means that a pixel class is a collection of pixels ![equation](https://github.com/diloc/Color_correction/blob/main/equations/pixelClass.png) within the same spatial neighborhood and similar color values. The pixel value $ z_ki $ is a random variable at location $i=0,1,2,…,n$.
- The reflectance of a pixel class is a collection of reflectance $R_k={r_ki }$, where $r_ki $ is a random variable representing the reflectance at the location $i=0,1,2,…,n$. Two adjacent reflectance are independent of each other, and the joint probability of is given by $$p(r_ki,r_kj )=p(r_ki )p(r_kj )$$. Based on the same assumption, all reflectance in a pixel class are independent events with joint probability $$p(R_k )=∏_(i=1)^n▒p(r_ki )$$.
- The illuminant of a pixel class is a collection of illuminants $L_k={l_ki }$. However, it is assumed that the illuminant is constant for all pixels in a class, meaning,  $$l_k=l_ki and L_k={l_k }$$. Then, the probability distribution of the illuminant is uniform, $p(l_k )=u_k$, being $u_k$ a constant value. 
- The illumination and the reflectance are statistically independent of each other $p(L_k,R_k )=p(L_k )p(R_k )$.
- The value of a pixel $z_ki$ is a function of the reflectance $r_ki$, the illuminant l_ki and the Gaussian noise w_ki with a mean equal to zero and variance $σ_k^2$ (Eq. 2). <br/>

![equation](http://www.sciweavers.org/tex2img.php?eq=1%2Bsin%28mc%5E2%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)

$z_ki=l_k r_ki+w_ki$	Eq. 2 <br/>

The multivariable function described in Eq. 2 can be statistically represented using the likelihood function. It is equivalent to Gaussian noise probability distribution (Eq. 3). <br/>

p(z_ki│l_k r_ki )=1/√(2π σ_k^2 )  exp⁡[-1/2  (z_ki-l_k r_ki )^2/(σ_k^2 )]	 Eq. 3 <br/>

- An observed image is divided into areas that correspond to individual pot areas.
- The objects of a pot area are segmented, such as leaves and soil.
- A pixel class is assigned individually to a segmented object; for instance, the plant pixel class contains the leaves, stem, and other plant organs. A class is a collection of $n$ pixels  $Z={z_i }$, where $z_i$ is a random variable representing the pixel value at location $i=0,1,2,…,n$.
- The reflectance of a pixel class is a collection of reflectances $R={r_i }$, where r_i is a random variable representing the reflectance at the location $i=0,1,2,…,n$. Reflectances are independent to each other such as $p(r_i,r_j )=p(r_i )p(r_j )$. Based on the independence assumption, we have $p(R)= \sum_{i=1}^{n} p(r_i )$ 
- The illumination $(l)$ has a uniform probability distribution over a pixel class $p(L)=u$ where $u$ is constant. 
- The illumination and the reflectance are statistically independent of each other $p(L,R)=p(L)p(R)$.
- A pixel value z_i is a function of the reflectance r_i, illuminant l and additive Gaussian noise w_i. This noise has a mean equal to zero and a standard deviation σ. <br/>

\begin{equation}
z_i=lr_i+w_i
\end{equation}
$z_i=lr_i+w_i$  (Eq. 1).


Before estimating the unknown illuminant, it is necessary to define the following assumptions:


### Likelihood: 
The likelihood of the pixel class is given the illuminant and reflectance and follows a normal distribution. <br/>
$$p(Z│L,R)= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma^2}}  exp⁡\biggl(-\frac{(z_i-lr_i)^2}{2\sigma^2}\biggr)$$  


### Priors: Reflectance & Illuminant: 
We created an **image dataset** to get the reflectance and illuminant prior distributions. It has images of green fabric pieces on pots and Macbeth colorChecker charts. They were illuminated using D65 standard illuminant.. <br/>

$$P(R)= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \tau^2}}  exp⁡\biggl(-\frac{(r_i-\mu)^2}{2\tau^2}\biggr)$$ <br/>
As the illumation is uniform over a pixel class the probability distribution is given by:
$$p(L)=u$$ <br/>

### Posterior
It is possible to analytically calculate the posterior distribution using the Bayes' rule as the prior is a conjugate prior for the likelihood function. The posterior distribution is given by:
$$P(L|Z)=\prod_{i=1}^{n} \int \frac{1}{\sqrt{2\pi \sigma^2}}  exp⁡\biggl(-\frac{(z_i-lr_i)^2}{2\sigma^2}\biggr) \frac{1}{\sqrt{2\pi \tau^2}}  exp⁡\biggl(-\frac{(r_i-\mu)^2}{2\tau^2}\biggr) d r_i $$ <br/>


### Maximum a posteriori 
We estimate the illumination value when the posterior distribution reaches the highest value.

$$  \hat{l}_{MAP} =\underset{l}{\operatorname{argmax}}  P(L|Z) = \frac{\sum_{i=1}^{n}z_i} {n \mu} $$

## Resources


* Source Repository: : 

## Dependencies
* Python (3. 7 or higher).
* Pandas (1.0.3 or higher).
* OpenCV (4.2.0 or higher).
* Datetime
* Scipy (1.4.1 or higher).
* Matplotlib (1.18.1 or higher).



# Results
The ColorBayes algorithm improved the accuracy of plant color on images taken by an indoor plant phenotyping system. Compared with existing approaches, it gave the most accurate metric results when correcting images from a dataset of Arabidopsis thaliana images.




