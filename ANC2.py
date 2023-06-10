import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def initialisation(n0, n1, n2):
    
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)
    
    parametres = {
        'W1' : W1,
        'W2' : W2,
        'b1' : b1,
        'b2' : b2
    }
    
    return parametres

def forward_propagation(X, parametres):
    #parametres    parametres
    W1 = parametres['W1']    
    W2 = parametres['W2']
    b1 = parametres['b1']
    b2 = parametres['b2']
    #parametres    parametres
    
    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    
    activations = {
        'A1' : A1,
        'A2' : A2
    }
    return activations

def log_loss(y,A):
    epsilon=1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def back_propagation(X, y, activations, parametres):
    
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']
    
    m = y.shape[1]
    
    dZ2 =A2 - y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims= True )
    
    dZ1 =np.dot(W2.T, dZ2) * A1 * (1- A1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims= True )
    
    gradients = {
        'dW1' : dW1,
        'dW2' : dW2,
        'db1' : db1,
        'db2' : db2
    }
    
    
    return (gradients)

def update(parametres, gradients, learning_rate):
    W1 = parametres['W1']
    W2 = parametres['W2']
    b1 = parametres['b1']
    b2 = parametres['b2']
    
    dW1 = gradients['dW1']
    dW2 = gradients['dW2']
    db1 = gradients['db1']
    db2 = gradients['db2']
    
    
    W1= W1 - learning_rate * dW1
    b1= b1 - learning_rate * db1
    W2= W2 - learning_rate * dW2
    b2= b2 - learning_rate * db2
    
    parametres = {
        'W1' : W1,
        'W2' : W2,
        'b1' : b1,
        'b2' : b2
    }
    
    return (parametres)

def predict(X, parametres):
    activation = forward_propagation(X, parametres)
    A2 = activation['A2']
    return(A2 >= 0.5)

def artificiel_neuronS(X_train , y_train, n1,learning_rate , i_max, test):
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
     
    parametres = initialisation(n0, n1, n2)
    
    train_loss=[]      
    train_acc=[]

    
    
    for i in tqdm(range(i_max), ncols = 100, desc ="Loading") :
        activations =forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, activations, parametres)
        parametres = update(parametres, gradients, learning_rate)
 
        
        
        #COURBE     #COURBE     #COURBE
        if i%100==0:
            
            #train
            train_loss.append(log_loss(y_train,activations['A2'])) 
            y_pred= predict(X_train, parametres)
            train_acc.append(accuracy_score(y_train.flatten(),y_pred.flatten()))
 
            
            
        #COURBE     #COURBE     #COURBE
        
        para = {
        'parametres' : parametres,
        'train_loss' : train_loss,
        'train_acc' : train_acc
    }
        
    
    y_pred = predict(X_train, parametres)
    print("3 parametres possible 'parametres', 'train_loss', 'train_acc'. ")
    
    #AFichage    #AFichage     #AFichage     
    if(test):
        plt.figure(figsize=(12,4))
    
        plt.subplot(1,2,1)
        plt.plot(train_loss)
        plt.title("log_loss graph")
    
        plt.subplot(1,2,2)
        plt.plot(train_acc) 
        plt.title("accuracy graph")
        plt.show()

   
 
    #AFichage   #AFichage    #AFichage
    return (para)

