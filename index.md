
# Deep Learning (CAS machine intelligence, 2023) 

This course in deep learning focuses on practical aspects of deep learning. We therefore provide jupyter notebooks ([complete overview of all notebooks](https://github.com/tensorchiefs/dl_course_2023/tree/master/notebooks) used in the course). 

For doing the hands-on part we recommend to use colab (you might need a google account) an internet connections is also needed. If you want to do it without internet connection on your own computer you can install anaconda ([details and installation instruction](anaconda.md)). Please note that we are not experts in anaconda and thus can only give limited support.

To easily follow the course please make sure that you are familiar with the some [basic math and python skills](prerequistites.md). 

## Info for the projects
You can join together in small groups and choose a topic for your DL project. You should prepare a poster and a spotlight talk (5 minutes) which you will present on the last course day. To get some hints how to create a good poster you can check out the links that are provided in <a href="https://www.dropbox.com/s/u1f6mqk4pc3uhxe/poster-guidelines.pdf?dl=1">poster_guidelines.pdf</a> 

If you need free GPU resources, we might want to follow the [instructions how to use google colab](co.md).  

Examples for projects from previous versions the DL course:
  [2018, 2019](projects.md)
  [2020](https://docs.google.com/spreadsheets/d/1NXinRQMifg_QNQs1fyn5HeiZNRnTGnIy1W7-ij-jQhg/edit?usp=sharing)
  [2021](https://docs.google.com/spreadsheets/d/18VFrPbKq3YSOg8Ebc1q1wGgkfgaWl7IkcCClGEDGj6Q/edit#gid=0)
  [2023](https://docs.google.com/spreadsheets/d/1TZf5hKekzOlBC7J0-EAltGOMTuZyrDhHu3ANve0q6H4/edit#gid=0)



**Fill in the Title and the Topic of your Projects until End of Week 5 [here](https://docs.google.com/spreadsheets/d/1d1y-Qf9OW7Vg30WzWwCckYPBMyRcg-d-qLG_lA0Z5jk/edit?usp=sharing)**

## Other resources 
We took inspiration (and sometimes slides / figures) from the following resources.

* Probabilistic Deep Learning (DL-Book) [Probabilistic Deep Learning](https://www.manning.com/books/probabilistic-deep-learning?a_aid=probabilistic_deep_learning&a_bid=78e55885). This book is written by us the tensorchiefs and covers the increasingly popular probabilistic approach to deep learning.

* Deep Learning Book (DL-Book) [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/). This is a quite comprehensive book which goes far beyond the scope of this course. 

* Convolutional Neural Networks for Visual Recognition [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/), has additional material and [youtube videos of the lectures](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC). While the focus is on computer vision, it also treats other topics such as optimization, backpropagation and RNNs. Lecture notes can be found at [http://cs231n.github.io/](http://cs231n.github.io/).

* More TensorFlow examples can be found at [dl_tutorial](https://github.com/oduerr/dl_tutorial/tree/master/tensorflow/) 

* Another applied course in DL: [TensorFlow and Deep Learning without a PhD](https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd)

## Dates 
The course is split in 8 sessions, each 4 lectures long. 

| Day  |      Date    |      Time    |
|:--------:|:--------------|:---------------|
| 1        | 21.02.2023 | 13:30-17:00 |
| 2        | 28.02.2023 | 13:30-17:00 |
| 3        | 07.03.2023 | 13:30-17:00 |
| 4        | 14.03.2023 | 13:30-17:00 |
| 5        | 21.03.2023 | 13:30-17:00 |
| 6        | 28.03.2023 | 13:30-17:00 |
| 7        | 04.04.2023 | 13:30-17:00 |
| 8        | 11.04.2023 | 13:30-17:00 |



## Syllabus (might change during course) 
- Day 1
  - Topics: Introduction, Fully Connected Networks (fcNN) 
  - Slides: [01_Introduction](https://github.com/tensorchiefs/dl_course_2023/blob/master/slides/01_Introduction.pdf)
  - Additional Material: 
    - [Network Playground](https://playground.tensorflow.org/)
  - Exercises and Homework: 
    - [01_simple_forward_pass](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/01_simple_forward_pass.ipynb), [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/01_simple_forward_pass.ipynb) 
  - Solutions to Exercises: 
    - [01_sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/01_simple_forward_pass_sol.ipynb), [01_sol_colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/01_simple_forward_pass_sol.ipynb) 



- Day 2
  - Topics: Looking back at fcNN, DL framework Keras, convolutional neural networks (CNN)
  - Slides:  [02_fcNN_CNN](https://github.com/tensorchiefs/dl_course_2023/blob/master/slides/02_fcNN_CNN.pdf)
  - Additional Material: 
    - [Understanding convolution](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
  - Exercises and Homework: 
    - [02_fcnn_with_banknote](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/02_fcnn_with_banknote.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/02_fcnn_with_banknote.ipynb)
    - [03_fcnn_mnist](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/03_fcnn_mnist.ipynb)  [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/03_fcnn_mnist.ipynb)
    - [04_fcnn_mnist_shuffled](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/04_fcnn_mnist_shuffled.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/04_fcnn_mnist_shuffled.ipynb) 
    - [05_cnn_edge_lover](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/05_cnn_edge_lover.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/05_cnn_edge_lover.ipynb) 
    
  <!--- auskommentieren  - [07_cifar10_norm](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/07_cifar10_norm.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/07_cifar10_norm.ipynb)--->
  - Solutions to Exercises: 
    - [02_sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/02_fcnn_with_banknote_sol.ipynb), [02_sol_colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/02_fcnn_with_banknote_sol.ipynb)
    - [03_sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/03_fcnn_mnist_sol.ipynb), [03_sol_colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/03_fcnn_mnist_sol.ipynb) 
    - [04_sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/04_fcnn_mnist_shuffled_sol.ipynb), [04_sol_colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/04_fcnn_mnist_shuffled_sol.ipynb) 
    - [05_sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/05_cnn_edge_lover_sol.ipynb), [05_sol_colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/05_cnn_edge_lover_sol.ipynb) 
   

- Day 3
  - Topics: Convolutional neural networks (CNN) 
  - Slides: [03_CNN](https://github.com/tensorchiefs/dl_course_2023/blob/master/slides/03_CNN.pdf)
  - Exercises and Homework:  
    - [06_cnn_mnist_shuffled](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/06_cnn_mnist_shuffled.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/06_cnn_mnist_shuffled.ipynb)
    - [07_cifar10_tricks](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/07_cifar10_tricks_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/07_cifar10_tricks_sol.ipynb)
	- [08_classification_transfer_learning_few_labels](https://github.com/tensorchiefs/dl_course_2023/blob/main/notebooks/08_classification_transfer_learining_few_labels.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/main/notebooks/08_classification_transfer_learining_few_labels.ipynb) 
    
  <!--- auskommentieren	- [09_1DConv](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/09_1DConv.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/09_1DConv.ipynb) --->
  - Solutions to Exercises:  
    - [06_sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/06_cnn_mnist_shuffled_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/06_cnn_mnist_shuffled_sol.ipynb) 
	- [08_sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/08_classification_transfer_learning_few_labels_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/main/notebooks/08_classification_transfer_learning_few_labels_sol.ipynb)
<!--- auskommentieren	- [09_sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/09_1DConv_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/09_1DConv_sol.ipynb) --->


- Day 4
  - Topics: Details Backpropagation in DL, MaxLike-Principle
  - Slides: [04_Training_Details](https://github.com/tensorchiefs/dl_course_2023/blob/master/slides/04_Details.pdf)
  - Exercises and Homework:
	- Backprop. linear regression eager [10_linreg_tensorflow](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/10_linreg_tensorflow.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/10_linreg_tensorflow.ipynb) 	
	- Optional Backprop static graph[11_backpropagation](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/11_backpropagation.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/11_backpropagation.ipynb)
	- Simple Max Like [12_maxlike](https://github.com/tensorchiefs/dl_book/blob/master/chapter_04/nb_ch04_01.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_book/blob/master/chapter_04/nb_ch04_01.ipynb)
	- Max Like MNIST [12b_mnist_loglike](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/12b_mnist_loglike.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/12b_mnist_loglike.ipynb)


- Day 5
  - Topics: Probabilistic Prediction Models
  - Slides:  [05_Probabilistic_Models](https://github.com/tensorchiefs/dl_course_2023/blob/master/slides/05_Probabilistic_Modeling.pdf)
  - Exercises and Homework:  
	- Linear Regression with Tensorflow Probability [13_linreg_with_tfp](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/13_linreg_with_tfp.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/13_linreg_with_tfp.ipynb)  
	- Poisson Regression with Tensorflow Probability [14_poisreg_with_tfp](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/14_poisreg_with_tfp.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/14_poisreg_with_tfp.ipynb)  
  - Solutions to Exercises:  
	- [13_Sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/13_linreg_with_tfp_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/13_linreg_with_tfp_sol.ipynb)   	
	- [14_Sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/14_poisreg_with_tfp_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/14_poisreg_with_tfp_sol.ipynb)   


- Day 6
  - Topics: Flexible CPDs
  - Slides: [06_flexible_CPDs](https://github.com/tensorchiefs/dl_course_2023/blob/master/slides/06_flexible_CPDs.pdf) 
  - Exercises and Homework:  
	- Regression with Tensorflow Probability on Images [15_faces_regression](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/15_faces_regression.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/15_faces_regression.ipynb)  
	- Visualization of Network Decisions with GradCam [16_elephant_in_the_room](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/16_elephant_in_the_room.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/16_elephant_in_the_room.ipynb)  
  - Solutions to Exercises:  
	- [15_Sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/15_faces_regression_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/15_faces_regression_sol.ipynb)   	
	- [16_Sol](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/16_elephant_in_the_room_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/16_elephant_in_the_room_sol.ipynb)  


- Day 7
  - Topics: Ensembling and Bayes
  - Slides: [07_ensembling_bayes](https://github.com/tensorchiefs/dl_course_2023/blob/master/slides/07_ensembling_bayes.pdf) 
  - Exercises and Homework:  
  	- Classification with Ensembles and Bayes [17_cifar10_ensemble_bayes](https://github.com/tensorchiefs/dl_course_2023/blob/master/notebooks/17_cifar10_classification_mc_and_vi_sol.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_course_2023/blob/master/notebooks/17_cifar10_classification_mc_and_vi_sol.ipynb)  
	- Bayesian Model for Coin Toss [18_bayesian_coin_toss](https://github.com/tensorchiefs/dl_book/blob/master/chapter_07/nb_ch07_03.ipynb) [colab](https://colab.research.google.com/github/tensorchiefs/dl_book/blob/master/chapter_07/nb_ch07_03.ipynb)

- Day 8
  - Topics: Bayes (continued), Projects
  - Slides: [08_bayes_2023](https://github.com/tensorchiefs/dl_course_2023/blob/master/slides/08_bayes_2023.pdf)  



