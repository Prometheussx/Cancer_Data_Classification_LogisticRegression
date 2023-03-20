# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:02:31 2023

@author: erdem
"""

# %% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Read Csv
data = pd.read_csv("Cancer_Data.csv")
print(data.info())
data.drop(["Unnamed: 32","id"],axis=1,inplace = True) #unnamed ve id sütun olarak sildik sütun sildiğimiz için axis 1 yaptık inplace true ise son durumu dataya kaydetmek için
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis] # iyi yada kötü olan tümörleri belirten m ve b harflerini obje olması sebebi ile sınıflandırmada kullanamazdık bu sebeple temel bir kod kalıbı ile 1 ve 0 çevirdik
print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis = 1)
# %% Normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values #büyük yada küçük syaıların ortralama durumu ekarte etmemesi iç,n yapılan bir işlemdir sayıları 0 ve 1 arasına sııştırır
#her sütunun her satırına tek tek uygular
# (x- min(x))/(max(x)-min(x)) normalize formülü

# %% Train Test Split %80 train %20 test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) #x ve y değerlerini al test olarak %20 yi ayır geriye kalanı train yap dedik
#random state 42 demek datayı train teste bölerken rastgele böldüğü için kütüphane her run edilişte farklı data elde edilmesin diye kullanılan datayı train ve testi yani 42 sayısına indexliyor denebilir nbu sayede random satate 42 dendiği her seferde aynı şekilde bölünmüş olucaktır

# normalde 30 adet özelik 455 adet canser hücresi şeklinde yazılması gerekirken(30,455) (455,30) yazıldığı için düzenlenmesi amacıyla ters çevirdik
x_train = x_train.T #transpozunu aldık yani matriksdeki nxm yi mxn yaptık
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

#%% Parameters initialize and sigmoid func.
#dimension =30
def initialize_weights_and_bias(dimension):

    w=np.full((dimension,1),0.01) #•dimension 30 olduğu için 30 a 1 lik 0.01 lerden oluşan w isimli ağırlıklar oluşturacak
    b = 0.0 #önerilen sabit sayı (bias)
    
    return w,b

#w,b = initialize_weights_and_bias(30)

def sigmoid(z): #bu yapı elde edilen deeğri 0 ve 1 arasına indirgiyerek bize 0 ve 1 arasında y_head değeri verir y_head değeri ise tahmin değeridir
    y_head = 1/(1+np.exp(-z)) #np.exp üssel ifade demek
    return y_head

#%%Forward backward propagation
def forward_bacward_propagation(w,b,x_train,y_train):
#matriks çarpımında (30 455) (30 1) çarpılmaz bu yüzden w'nin transpozu alınır
    #FORWARD
    z = np.dot(w.T,x_train)+b # w  matriksinin transpozu ile x_train matrixi çarğpılır np.dot budur sonuçdata bias sabiti eklenir
    y_head = sigmoid(z) #tahmin verisi elde edilir
    loss= -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) #modeldeki kayıp oranını hesaplar
    cost = (np.sum(loss))/x_train.shape[1] #toplam kaybı 455 bölerek net kayıp bulunur

    #BACKWARD
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] #ağırlığın türevini aldık tahmin sonuç arasındaki farkın transpozuyla değerleri çarpıp 455 bölüyoruz
    derivative_bias= np.sum(y_head-y_train)/x_train.shape[1] #biasın türevini aldık
    #burada bias ve ağırlıktaki ortalama değişimi bulmuş olduk slope bulduk yani 
    gradients = {"derivative_weight": derivative_weight, "derivative_bias":derivative_bias} #parametrelerin depolandığı sözlük yapısı
    return cost,gradients



#%% Updating(Learning) Parameters
def update(w, b, x_train, y_train, learning_rate, number_of_iteration): #number of iteration kaç tur denenceğini söyler deneme yanımayla bulunur
    cost_list =[]
    cost_list2 = []
    index =  []
    
    
    #updating(Learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
       # make forward and backward propagation and find cost and gradients
       cost,gradients = forward_bacward_propagation(w,b,x_train,y_train)
       cost_list.append(cost) #ilk cost değerimizi tutuyoruz
       #lets update 
       #(w = w-a*türev w)  (b = b-a*türev b) bu sayede yeni değerler elde ederek costu minimize edene kadar devam ederiz en minimum değerde mükemmel eğitimi vermiş oluruz a=learning_rate öğrenme hjızı
       w = w - learning_rate * gradients["derivative_weight"]   
       b = b - learning_rate * gradients["derivative_bias"]   
       if i % 10 == 0: #her 10 turda 1 costu kaydetmemiz için yapılır  daha hızlı ve basit incelenbilmesi için kullanılır 10 keyfi seçilmiştir
            cost_list2.append(cost) #her 10 turdaki 1 costu kaydeder
            index.append(i)
            print("Cost after iteration %i: %f" %(i,cost)) #hataları bastırrı
            
      # we update(learn) parameters wights and bias
    parameters = {"weight": w, "bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation="vertical")#x ekseninde yer alıcak isimleri belirttim rotasyon olarak dikey şekilde
    plt.xlabel("Number of İteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
      
#%% predict
#30 özelliğimiz 114 kanserimiz var
def predict(w,b,x_test):
     #x_test is a input for forward propagation
     z = sigmoid(np.dot(w.T,x_test)+b)
     Y_prediction = np.zeros((1,x_test.shape[1])) #1 e 114 lük 0 lardan oluşan boş matriks oluşturduk
     #if z is bigger than 0.5 our prediction is sign one (y_head = 1)
     #if z is smaller than 0.5 our prediction is sign one (y_head = 0)
     for i in range(z.shape[1]):
         if z[0,i]<= 0.5: #tek boyutlu matrikste tek tek değer denemesi yapıyor 0. sütun 1. 2. 3. 4. değerleri gibi
             Y_prediction[0,i] = 0
         else:
             Y_prediction[0,i] = 1
             
     return Y_prediction



#%% Logistic Regressin
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)  #predict etme 
  
    # Print train/test Errors
    #gerçek 1 tahmin 0 sa mmutlak değerden 1 çıkar 1*100 den 100 çıkar 100-100 den 0 çıkar ve sonuç hatalı dmeketir 
    #tahminde sonuçta 0 ve 0 sa 0-0 dan 0 100*0 dan 0 100-0 dan 100 çıkar ve sonuç doğrudur
 
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300) #çalıştırma kodu










