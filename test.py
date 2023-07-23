from client import Client
import sys
import time

client = Client("192.168.178.128")
print("Client created")
while True:
    print(client.look())
    time.sleep(1)
