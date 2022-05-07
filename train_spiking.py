#All library
from neko.backend import pytorch_backend as backend
from neko.datasets import MNIST
from neko.evaluator import Evaluator
from neko.layers import ALIFRNNModel
from neko.learning_rules import Eprop
from neko.trainers import Trainer
import matplotlib.pyplot as plt
from encoding import encoder
import numpy as np
from sklearn.model_selection import train_test_split

#Dataset
x_train, y_train, x_test, y_test = MNIST().load()
x_train, x_test = x_train / 255., x_test / 255.

Y_sum=np.sum(y_train[:,:2],axis=1)
Y_train=y_train[:,:2][Y_sum>0]
X_train=x_train[Y_sum>0]

_, x_data, _, y_data = train_test_split(X_train, Y_train, test_size=0.50, random_state=42)
print("Data loaded")

#Training Dataset
enc_train=encoder()
enc_train.create_Encoded_Dataset(x_data)
x_train_enc=enc_train.x_enc
print("Data encoded")

# #Testing Dataset
# enc_test=encoder()
# enc_test.create_Encoded_Dataset(x_test)
# x_test_enc=enc_test.x_enc

#Create the model
model = ALIFRNNModel(128, 2, backend=backend, task_type='classification', return_sequence=False)
evaluated_model = Evaluator(model=model, loss='categorical_crossentropy', metrics=['accuracy', 'firing_rate'])
algo = Eprop(evaluated_model, mode='symmetric')
trainer = Trainer(algo)
print("Model created")
X=np.array(x_train_enc).astype(np.float32)
#Train the model
trainer.train(X, y_data, epochs=100)