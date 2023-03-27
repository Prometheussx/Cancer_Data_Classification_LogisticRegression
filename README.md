# 🦠 Cancer Data Classification (Logistic Regression)
## 📑 Project Summary
In this project, I have used logistic regression to classify the cancers we have as benign or malignant.       
No machine learning libraries were used in this project pytorch sckitlearn etc.

## ⚙️ Data information
My data consists of 569 cancer cells and 30 characteristics of each cell.

## 📈 Plot of Cost and Number of Iterations
We examine how much or how little error the model we obtained with the graphical design of the cost and Number of Iterations ratios makes. As can be seen here, the margin of error of the model is very, very low.

![Ekran görüntüsü 2023-03-20 235346](https://user-images.githubusercontent.com/54312783/226463544-3fcad5f4-99f1-42a9-8422-a74093099d34.png)

## 🗂 Result cost values and accuracy values 
0.692977 at cost We have reached 0.152266 cost value after 290 updates. As a result of this situation, the model has reached the number of turns we want and has reached the test accuracy: 96.49122807017544 %, proving that it is a healthy model. As noticed, as the accuracy value approached 100, the cost value started to decrease less.

**For example:**

Cost after iteration 260: 0.158155

Cost after iteration 270: 0.156091

Cost after iteration 280: 0.154131

Cost after iteration 290: 0.152266

From round 260 to round 290 there was only an average decrease of 0.006 This shows that the error rate decreases to very minimal values as the accuracy rate increases, as seen in the graph. In this case, the training is finished and the result is considered sufficient because further training does not make a great contribution to the model.

## 🔗 Kaggle Link
 https://www.kaggle.com/code/erdemtaha/detection-cancer-with-logistic-regression/settings?scriptVersionId=123550989
## 🧾 General note
The values used in the project, such as lerning_rate and num_iterations, were determined according to the accuracy and graph type of the results.


![Ekran görüntüsü 2023-03-20 235844](https://user-images.githubusercontent.com/54312783/226464805-f28f2ce1-3652-426a-804d-b39671ecb899.png)
