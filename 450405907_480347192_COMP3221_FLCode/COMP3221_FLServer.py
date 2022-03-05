from _thread import *
import threading
import _thread
import time
import socket
import pickle
import sys
import json
import copy
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# It will use local IP to communicate
IP = "127.0.0.1"
GLOBAL_ITERATION = 100
INITIAL_WAIT = 30 # It will first wait initially 30 seconds for each of the server to connect
HEADER = 64
TOTAL_CLIENT = 5
FORMAT = "utf-8"

class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Server:
    def __init__(self, portNo, subClient):
        self.portNo = portNo
        self.subClient = subClient
        self.noClient = 0
        self.client = {}
        self.client_model = {}
        self.lock = threading.Lock()
        self.initial_lock = threading.Lock()
        self.fed_lock = threading.Lock()
        self.barrier = None
    
        self.l_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.l_socket.bind((IP, self.portNo))
        self.l_socket.listen(5)

        self.server_model = MCLR()
        self.total_train_sample = 0

        self.initial_connection = True
        self.fed_avg_started = False
        self.server_printed = False

    def send_parameters(self, s):

        msg = self.server_model   
        msg = pickle.dumps(msg)
        length = sys.getsizeof(msg)

        # sendLength = json.dumps(length)
        # print(sys.getsizeof(sendLength))
        # sendLength = str.encode(sendLength)

        sendLength = str(length).encode(FORMAT)
        sendLength += b' ' * (HEADER - len(sendLength))
        # sendLength = str(length).encode(FORMAT)
        # sendLength += b' ' * (HEADER - len(sendLength))

        s.sendall(sendLength)

        msg = bytes(msg)
        s.sendall(msg)

        return

    def aggregate_parameters(self, subclient):

        a = random.randint(1, 5) 
        b = random.randint(1, 5)

        while (a == b):
            b = random.randint(1, 5)

        print("Aggregating new global model")
        for param in self.server_model.parameters():
            param.data = torch.zeros_like(param.data)
        
        i = 1

        for key, user in self.client_model.items():
            if(subclient == 1):
                if(not(i == a or i == b)):
                    i += 1
                    continue
            for server_param, user_param in zip(self.server_model.parameters(), user.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * self.client[key] / self.total_train_sample
            i += 1
            

        return self.server_model    

    def recv_model(self, c, client):

        time.sleep(0.5)
        
        data_rev = c.recv(HEADER)
        data_rev = data_rev.decode(FORMAT)
        # data_rev = c.recv(64)
        # data_rev = json.loads(data_rev)
        while not data_rev:
            time.sleep(1)
            data_rev = c.recv(HEADER)
            print(data_rev)
            data_rev = data_rev.decode(FORMAT)

        length = int(data_rev)

        time.sleep(0.5)

        data = c.recv(length)
        model = pickle.loads(data)

        print("Recieving Model from " + client)
        self.fed_lock.acquire()
        self.client_model[client] = model
        self.fed_lock.release()
        return

    def fed_server(self, c, addr, s, client):
        try:    
            #If it's a initial connection, then we should wait for thirty seconds to begin the operation.
            self.initial_lock.acquire() #The first thread to hold this lock will keep other threads from going on.

            if(self.initial_connection):
                time.sleep(INITIAL_WAIT)
                self.initial_connection = False
                self.fed_avg_started = True
                print(self.client)
                self.barrier = threading.Barrier(self.noClient)

            self.initial_lock.release()

            i = self.barrier.wait()

            # self.fed_helper(c, addr, s, client)
            
            for k in range(1, GLOBAL_ITERATION + 1):

                i = self.barrier.wait() 

                if(i == 0):
                    print("Global Iteration " + str(k))
                    print("Total number of client " + str(self.noClient))
                    
                i = self.barrier.wait()
        
                self.send_parameters(s)

                i = self.barrier.wait()

                self.recv_model(c, client)

                i = self.barrier.wait()

                if(i == 0):
                    self.aggregate_parameters(self.subClient) 
                    self.client_model = {}
                    print("Broadcasting new global model") 
                    print("")
                    print("===============================================")
                    print("")

                i = self.barrier.wait()
        
            s.close()
            c.close()
            return
        except Exception as e:
            print(e)
            s.close()
            c.close()
        return

    def initial_listen(self, c, addr):
        try:
            while self.fed_avg_started: # To make sure not to get further connection after fed algorithm started
                time.sleep(1)

            #First recive the information
            data_rev = c.recv(1024)
            data_rev = data_rev.decode()
            handShakingMSG = json.loads(data_rev)
            table = handShakingMSG.split(",")
            clientPortNo = int("600" + str(table[0][-1]))

            self.lock.acquire()

            self.total_train_sample += int(table[1])
            self.client[str(table[0])] = int(table[1])
            self.noClient += 1

            self.lock.release()

            print(clientPortNo)

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: #s = socket.socket()         # Create a socket object
                s.connect((IP, clientPortNo))
                self.fed_server(c, addr, s, str(table[0]))
  
        except Exception as e:
            print(e)

    def listen(self):
        try:
            while True:

                if(self.noClient == TOTAL_CLIENT): #Total number of the client
                    time.sleep(10)
                    continue

                c, addr = self.l_socket.accept()
                _thread.start_new_thread(self.initial_listen,(c, addr))
                
        except Exception as e:
            print(e)
      
if __name__ == "__main__":
    try:
        portNo = int(sys.argv[1])
        subClient = int(sys.argv[2])
        server = Server(portNo, subClient)
        server.listen()

    except Exception as e:
        print(e)