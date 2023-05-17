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
- An observed image $(5105 x 3075 pixels)$ comprises three independent colour channels (c =R,G,B).
- The reflectance $r_cj$ is a random variable at the location $j=0,1,2,…,m$ , and colour channel c. Two adjacent reflectances are independent of each other, and the joint probability is given by $p(r_cj,r_cl )=p(r_cj )p(r_cl )$. Based on the same assumption, all reflectance are independent events with joint probability $\prod_{j=1}^{m} \ p(r_{cj})$.
- The illuminant $l_{cj}$ is also a random variable at the location $j=0,1,2,…,n$, and colour channel $c$.
- The illumination and the reflectance are statistically independent of each other $p(L_cj,R_cj )=p(L_cj )p(R_cj )$.
- An image is divided into an m number of small images which correspond to individual pot areas A_p where the index $p=0,1,2,…,m$ indicates the number of pot areas. It means that each pot area $A_p$ has a predetermined n number of pixels $Z_{cp}=\left\{z_{cph}\right\}$ at the location $h=0,1,2,…,q$. Also, the reflectance $R_{cp}=\left\{z_{cph}\right\}$  and illuminant $L_{cp}=\left\{l_{cph}\right\}$ associated with each pixel within a pot area share the same location h.
- The illuminant is constant for all pixels within a pot area A_p, meaning,  $l_{cp}=l_{cph}$ and $L_{cp}=\left\{l_{cp}\right\}$. Then, the probability distribution of the illuminant is uniform, $p(l_{cp} )=u_{cp}$, where $u_{cp}$ is a constant value. However, two adjacent pot area illuminants are independent of each other $p(l_{cp},l_{cq})=p(l_{cp} )p(l_{cq} )$

### Likelihood: 
The likelihood of the pixel class is given the illuminant and reflectance and follows a normal distribution. <br/>
$$p(z_{cki}│l_{cki},r_{cki})= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma_{ck}^2}}  exp⁡\biggl(-\frac{(z_{cki}-l_{ck}r_{cki})^2}{2\sigma_{ck}^2}\biggr)$$  


### Priors: Reflectance & Illuminant: 
We created an **image dataset** to get the reflectance and illuminant prior distributions. It has images of green fabric pieces on pots and Macbeth colorChecker charts. They were illuminated using D65 standard illuminant.. <br/>

$$P(r_{cki})= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \tau_{ck}^2}}  exp⁡\biggl(-\frac{(r_i-\mu_{ck})^2}{2\tau_{ck}^2}\biggr)$$ <br/>
As the illumation is uniform over a pixel class the probability distribution is given by:
$$p(l_{ck})=u_{ck}$$ <br/>

### Posterior
It is possible to analytically calculate the posterior distribution using the Bayes' rule as the prior is a conjugate prior for the likelihood function. The posterior distribution is given by:
$$P(L_{ck}|Z_{ck})=\prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma_{ck}^2}}  exp⁡\biggl(-\frac{(z_{cki}-\mu_{ck}l_{ck})^2}{2\sigma_{ck}^2}\biggr) \frac{1}{\sqrt{2\pi \tau_{ck}^2}}  exp⁡\biggl(-\frac{(r_{cki}-\mu_{ck})^2}{2\tau_{ck}^2}\biggr) $$ <br/>


### Maximum a posteriori 
We estimate the illumination value when the posterior distribution reaches the highest value.

$$  \hat{l}_{MAP} =\underset{l_{ck}}{\operatorname{argmax}}  P(L_{ck}|Z_{ck})$$

## Resources


* Source Repository (prior): https://github.com/diloc/Color_correction/blob/8dea8b92ac3cea5e5c198348c04a50d10c2f8adb/Color_Constancy/prior.ipynb
* Source Repository (main): https://github.com/diloc/Color_correction/blob/919477408e1679f0c3c715a99ab9bff2afca433f/Color_Constancy/main.ipynb

## Dependencies
* Python (3. 7 or higher).
* Pandas (1.0.3 or higher).
* OpenCV (4.2.0 or higher).
* Datetime
* Scipy (1.4.1 or higher).
* Matplotlib (1.18.1 or higher).



# Results
The ColorBayes algorithm improved the accuracy of plant color on images taken by an indoor plant phenotyping system. Compared with existing approaches, it gave the most accurate metric results when correcting images from a dataset of Arabidopsis thaliana images.




