import numpy as np
import requests, gzip, os, hashlib


#fetch data
path='./data'
def fetch(url):
  fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
     with open(fp, "rb") as f:
        data = f.read()
  else:
     with open(fp, "wb") as f:
        data = requests.get(url).content
        f.write(data)
  return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

#Validation split
rand=np.arange(60000)
np.random.shuffle(rand)
train_no=rand[:50000]
val_no=np.setdiff1d(rand,train_no)
X_train,X_val=X[train_no,:,:],X[val_no,:,:]
Y_train,Y_val=Y[train_no],Y[val_no]


#Sigmoid and its derivative
def sigmoid(x):
  return 1/(np.exp(-x)+1)
def d_sigmoid(x):
  return (np.exp(-x))/((np.exp(-x)+1)**2)


#Softmax
def softmax(x):
  exp_element=np.exp(x-x.max())
  return exp_element/np.sum(exp_element,axis=0)
def d_softmax(x):
  exp_element=np.exp(x-x.max())
  return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))
   
#Initializing weights
def init(x,y):
  layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
  return layer.astype(np.float32)

np.random.seed(42)
l1=init(28*28,128)
l2=init(128,10)

#forward and backward pass
def forward_backward_pass(x,y):
  targets = np.zeros((len(y),10), np.float32)
  targets[range(targets.shape[0]),y] = 1

  x_l1p=x.dot(l1)
  x_sigmoid=sigmoid(x_l1p)
  x_l2p=x_sigmoid.dot(l2)
  out=softmax(x_l2p)

  error=2*(out-targets)/out.shape[0]*d_softmax(x_l2p)
  update_l2=x_sigmoid.T@error
    
    
  error=((l2).dot(error.T)).T*d_sigmoid(x_l1p)
  update_l1=x.T@error

  return out,update_l1,update_l2 

#training
epochs=10000
lr=0.001
batch=128

losses,accuries,val_accuracies=[],[],[]

for i in range(epochs):
    #randomize and create batches
    sample=np.random.randint(0,X_train.shape[0],size=(batch))
    x=X_train[sample].reshape((-1,28*28))
    y=Y_train[sample]

    out,update_l1,update_l2=forward_backward_pass(x,y)   
    category=np.argmax(out,axis=1)
    
    accuracy=(category==y).mean()
    accuries.append(accuracy.item())
    
    loss=((category-y)**2).mean()
    losses.append(loss.item())
    
    #SGD 
    l1=l1-lr*update_l1
    l2=l2-lr*update_l2


    #testing our model using the validation set every 20 epochs
    if(i%20==0):    
      X_val=X_val.reshape((-1,28*28))
      val_out=np.argmax(softmax(sigmoid(X_val.dot(l1)).dot(l2)),axis=1)
      val_acc=(val_out==Y_val).mean()
      val_accuracies.append(val_acc.item())
    if(i%1000==0): print(f'For {i}th epoch: train accuracy: {accuracy:.3f}| validation accuracy:{val_acc:.3f}')



test_out=np.argmax(softmax(sigmoid(X_test.dot(l1)).dot(l2)),axis=1)
test_acc=(test_out==Y_test).mean().item()
print(f'Test accuracy = {test_acc:.4f}')
np.savez('weights',l1,l2)
