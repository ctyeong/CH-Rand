# CH-Rand

*Channel Randomisation* (CH-Rand) is the technique to augment image data by randomising the RGB channels to encourage neural networks to learn normal compositions of "colour" in self-supervised manners. 
This repository is the official release of the codes used for the following preprint: 

*"Self-supervised Representation Learning for Reliable Robotic Monitoring of Fruit Anomalies", Taeyeong Choi, Owen Would, Adrian Salazar-Gomez, and Grzegorz Cielniak, Available at [\[arXiv:2109.10135\]](https://arxiv.org/abs/2109.10135).* 

As explained in the paper, CH-Rand has been designed to solve *fruit anomaly detection* problem in the [One-class Classification](https://en.wikipedia.org/wiki/One-class_classification) scenario, in which classifiers can only access the data of normal instances during training but must be able to identify anomalous instances in test. 
For self-supervised learning, CH-Rand is used to set up the pretext task to classify randomised images `x'=CHR(x)`, where `CHR` permutes the RGB channels in the normal image `x` with a possibility of repeatition (e.g., RRR, RRG, RRB, RGR, ..., BBB) &mdash; i.e., 26 possible `x'` exist excluding the original permutation RGB, as shown in [Examples](https://github.com/ctyeong/CH-Rand#examples) below.

After the proxy task, the learnt feature representations from a middle layer of the classifier can then be utilised to measure the degree of anomaly for tested image samples. To be specific, for each test input, the mean Euclidean distance to the *k* nearest neighbors in the training set is calculated as anomaly score supposing that anomalous images would tend to produce higher mean distances.


# Contents

1. [Examples](https://github.com/ctyeong/CH-Rand#examples)
1. [Installation](https://github.com/ctyeong/CH-Rand#installation)
1. [Training](https://github.com/ctyeong/CH-Rand#Training)
<!-- 1. [Random Splits](https://github.com/ctyeong/Riseholme-2021#random-splits)
1. [Benchmark Performance](https://github.com/ctyeong/Riseholme-2021#benchmark-performance) --> 
1. [Citation](https://github.com/ctyeong/CH-Rand#citation)
1. [Contact](https://github.com/ctyeong/CH-Rand#contact)  

# Examples 

Three examples are displayed below, in each of which the original RGB image of strawberry is followed by 26 possible channel randomised images. 

## Example 1
![](Figs/ex1_rgb.png)
| **26 Channel Randomisations**  | 
|--------------------|
| ![](Figs/ex1_rand.png)| 

## Example 2
![](Figs/ex2_rgb.png)
| **26 Channel Randomisations**  | 
|--------------------|
| ![](Figs/ex2_rand.png)| 

## Example 3
![](Figs/ex3_rgb.png)
| **26 Channel Randomisations**  | 
|--------------------|
| ![](Figs/ex3_rand.png)| 


# Installation

1. Clone the repository
    ```
    $ git clone https://github.com/ctyeong/CH-Rand.git
    ```

2. Install the required Python packages
    ```
    $ pip install -r requirements.txt
    ```
    - Python 3.8 is assumed to be installed already.


# Training

## Dataset
In this tutorial, we assume that [Riseholme-2021](https://github.com/ctyeong/Riseholme-2021) &mdash; the large image dataset for strawberry anomaly detection available [here](https://github.com/ctyeong/Riseholme-2021)
&mdash; has been cloned inside the root directory of `CH-Rand`. 
Even if a custom dataset is used, the current version of CH-Rand can be executed without major modification, as long as the same directory structure is adopted as in Riseholme-2021. 

## Config File
Prepare a configuration file with the extension of `.yaml` to pass required hyper parameters, such as data path, learning rate, image size, and extras. 
Use `Configs/config.yaml` as a template to provide your own parameters if necessary. Some of the parameters are explained below:

- `fc_feat`: Feature representations are extracted from the fully connected layer instead of the last convolutional layer during regular validations and tests.
- `epoch_size`: One epoch consists of `epoch_size` batch updates as opposed to the traditional concept, where all training samples are involved per epoch.
- `aug_mode`: One in {CH-Rand, CH-Perm, CH-Split} is usable, and read the paper above to learn how each works. 
- `portion`: [0., 1.] to determine proportionally how many pixels get affected by the predefined randomisation &mdash; i.e., 0: None and 1: All pixels. For 0<`portion`<1, pixels of similar intensities are selected once the input image has been converted to its grayscale version.
- `stop_criterion`: Training stops if the validation accuracy exceeds `val_acc_threshold` this number of times in a row.


# Citation
```
@article{CWSC21,
    title={Self-supervised Representation Learning for Reliable Robotic Monitoring of Fruit Anomalies}, 
    author={Taeyeong Choi and Owen Would and Adrian Salazar-Gomez and Grzegorz Cielniak},
    year={2021},
    journal={arXiv},
}
```


# Contact

If there is any questions about the dataset, please do not hesitate to drop an email to tchoi@lincoln.ac.uk or gcielniak@lincoln.ac.uk. Thanks!

