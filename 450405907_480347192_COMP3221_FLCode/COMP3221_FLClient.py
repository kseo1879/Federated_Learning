from _thread import *
import threading
import _thread
import time
import pickle
import socket
import sys
import json
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# It will use local IP to communicate
IP = "127.0.0.1"
SERVER_PORT = 6000
LEARNING_RATE = 0.01
BATCH_SIZE = 5
FORMAT = "utf-8"
HEADER = 64

def get_data(id=""):
    train_path = os.path.join("FLdata", "train", "mnist_train_" + str(id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_" + str(id) + ".json")
    train_data = {}
    test_data = {}

    with open(os.path.join(train_path), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    with open(os.path.join(test_path), "r") as f_test:
        test = json.load(f_test)
        test_data.update(test['user_data'])

    X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
    X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    train_samples, test_samples = len(y_train), len(y_test)
    return X_train, y_train, X_test, y_test, train_samples, test_samples

class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Client():

    def __init__(self, clientId, portNo, optMethod):
        self.id = clientId
        self.portNo = portNo
            
        self.l_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.l_socket.bind((IP, self.portNo))
        self.l_socket.listen(5)

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(clientId)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]

        if optMethod:
            batch_size = BATCH_SIZE
        else:
            batch_size = self.train_samples

        self.trainloader = DataLoader(self.train_data, batch_size)
        self.testloader = DataLoader(self.test_data, self.test_samples)
        self.loss = nn.NLLLoss()
        self.model = None
        self.optimizer = None
        # self.model = MCLR()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

        self.initial = True
        file_path = f"./{self.id}_log.txt"
        self.f = open(file_path, "w")

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
            
    def train(self, epochs):
        print("Local training...")
        LOSS = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        print(loss.data)
        self.f.write(str(loss.data))
        self.f.write("\n")
        return loss.data
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
            print("Accuracy of ",self.id, " is: ", test_acc*100, "%")
            msg = f"Accuracy of {self.id} is: {test_acc*100} %\n"
            self.f.write(msg)
        return test_acc
    
    def send_local_model(self, s):


        msg = self.model
        msg = pickle.dumps(msg)
        length = sys.getsizeof(msg)

        sendLength = str(length).encode(FORMAT)
        sendLength += b' ' * (HEADER - len(sendLength))

        # sendLength = json.dumps(length)
        # sendLength = str.encode(sendLength)


        s.sendall(sendLength)

        print("Sending new local model")
        print("")
        self.f.write("Sending new local model\n\n")


        msg = bytes(msg)
        s.sendall(msg)
        # length = (sys.getsizeof(msg))

        # msg_legth = sys.getsizeof(msg)
        # sendLength = str(msg_legth).encode(FORMAT)
        # sendLength += b' ' * (HEADER - len(sendLength))

        # print(sendLength)
        # s.send(sendLength)

        # print("Sending new local model")
        # msg = pickle.dumps(msg)
        # msg = bytes(msg)
        # s.sendall(msg)

        return 
    
    def recv_global_model(self, c):

        time.sleep(0.5)

        data_rev = c.recv(HEADER)
        data_rev = data_rev.decode(FORMAT)
        # data_rev = c.recv(64)
        # data_rev = json.loads(data_rev)
        
        while not data_rev:
            time.sleep(1)
            data_rev = c.recv(HEADER)
            data_rev = data_rev.decode(FORMAT)
       
        length = int(data_rev)

        time.sleep(0.5)

        print("Recieving new global model")
        self.f.write("Recieving new global model\n")

        data = c.recv(length)
        if not data:
            print("Didn't recieve the length")
        model = pickle.loads(data)

        return model

    def fed_client(self, c, addr, s):
        try:
            i = 0
            while True:

                print("")
                print("I am " + self.id)
                self.f.write("\nI am " + self.id + "\n")
                

                #first recieve the model from the server
                model = self.recv_global_model(c)

                if(self.initial):
                    self.model = copy.deepcopy(model)
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
                    self.initial = False

                self.set_parameters(model)
                
                self.train(1)

                self.test()

                self.send_local_model(s)

                i += 1

                if(i == 100):
                    break
            
            self.f.close()
            c.close()
            s.close()
            return      
        except Exception as e:
            print(e)
        return

    def hand_shake(self):

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: #s = socket.socket()         # Create a socket object
                #connecting to the server port
                s.connect((IP, SERVER_PORT))

                handShake = f"{self.id},{self.train_samples}"
                msg = json.dumps(handShake)
                msg = str.encode(msg)
                s.sendall(msg)
                
                time.sleep(0.1)
                c, addr = self.l_socket.accept()
                self.fed_client(c, addr, s)

        except Exception as e:
            print(e)

        return
        
if __name__ == "__main__":
    try:
        clientId = (sys.argv[1])
        portNo = int(sys.argv[2])
        optMethod = int(sys.argv[3])
        client = Client(clientId, portNo, optMethod)    #First initialize it's class
        client.hand_shake()

    except Exception as e:
        print(e)