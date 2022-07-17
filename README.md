# Deep-Learning-Case-Study
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#shallow-nn-model">Shallow NN model(2 layers)</a></li>
    <li><a href="#depp-nn-model">Deep NN model(L layers)</a></li>
  </ol>
</details>

# Shallow NN model
In this case, we are about to implement a 2-layers neural networks as a classifier to define regions as either red or blue. More specifically, there are 
many points(dots) in a plane, some of them are blue while the others are red. The dataset we used in this case can be visualized as Figure. 1 
(like a "flower"). Our goal is to build a model to fit this dataset as Figure. 2

Figure. 1 :![image](https://user-images.githubusercontent.com/77977624/179396611-7dcd76dc-14a1-4421-9caa-7255779fa8b1.png)

Figure. 2 :![Figure_2](https://user-images.githubusercontent.com/77977624/179397135-d4220ec7-c348-45f8-ad85-3d38703c693c.png)

Before implementing a nueral networks, we firstly check how logistic regression perform on this problem by running the logistic regression model we've
implemented previously. The outcome can be presented as Figure. 3, it turns out that logistic regression doesn't perform well. So let's see how about neural
networks perform

Figure. 3 :![Figure_4](https://user-images.githubusercontent.com/77977624/179399199-e6520667-580d-4903-9a7b-e2386267153e.png)

To solve this planar data classification problem, we are going to build a 2-layers neural networks which has only one hidden layer. The model structure is shown below:

![image](https://user-images.githubusercontent.com/77977624/179400025-33bec812-21b2-4076-a11c-d429b1865234.png)

Mathematically, we can represent the model as following formulas:

![image](https://user-images.githubusercontent.com/77977624/179400125-908f84f8-9300-49cb-a0a6-1eec3c9c8c35.png)

During the experiment, we set learning rate and number of iterations equals to 1.2 and 1000 respectively, while took radom initialization strategy to initialize
parameters w and b. After 1000 iterations, we end up with a accuracy at 90% and getting a well-classified solotion as Figure. 2
