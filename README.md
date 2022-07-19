# Deep-Learning-Case-Study
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#shallow-nn-model">Shallow NN model(2 layers)</a></li>
    <li><a href="#deep-neural-networks-model">Deep Neural Networks(DNN) model(L layers)</a></li>
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

# Deep Neural Networks Model
From previous experiment(details can be seen at https://github.com/zhuzhongju/Deep-Learning-Basics.git), we achieve a accuracy at 70% for Cat VS Non-Cat problem through implementing a logistic regression model as a classifier. In this case, we are going to build a deep nerual network to run on the same dataset again and compare these two models in terms of accuracy. To train such a model, we firstly need to preprocess the dataset(reshape, standarlize and so on). Here, we won't go through preprocessing stage(it is as same as the previous experiment, please figure it out by visiting the website.) and only foucs on DNN model's architecture and mathematical erxpression.

The simplified network representation of L-layers deep neural network is illustrated as below:

![image](https://user-images.githubusercontent.com/77977624/179466302-5b583cd0-4141-4cb2-bfda-f25825c508c7.png)

The model can be summarized as : [LINEAR --> RELU] X (L-1) --> LINEAR --> SIGMOID
It is noted that intermediate values in every layer, like 𝑍[𝐿], and 𝐴[𝐿], should be stored in the cache and passed to the next step to compute gradients through backward propagation. The figure below illustrates how those intermediate values affect the backward propagation module. The purple blocks represent the forward propagation, and the red blocks represent the backward propagation.

![image](https://user-images.githubusercontent.com/77977624/179688760-c34ebd29-3e99-48a3-8e37-624052129a73.png)

The formulas used in forward propagation are the same as is used in the shallow NN model case. The backward propagation module can be mathematically represented as:

![image](https://user-images.githubusercontent.com/77977624/179692517-93f86017-464f-433f-859f-5fdf21746ec3.png)

![image](https://user-images.githubusercontent.com/77977624/179692603-04c3145c-12fe-4139-927d-bc8a09d4e1cd.png)

After applying the DNN model on this probelm, we end up with a accuracy at 80%. Its layer dimension is [12288, 20, 7, 5, 1], which means it has 4 hidden layers and an output layer with 12288, 20, 7, 5, 1 units in each layer respsectively. The cost for every iteration is drawn below:

![cost-iterations](https://user-images.githubusercontent.com/77977624/179698331-47d18463-2cfb-406a-98fd-bfb7b44633ae.png)

