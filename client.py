import cv2
import io
import numpy as np
import sys
import socket
import struct
import threading
import torch
from PIL import Image
from threading import Thread
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform


class Client:
    def __init__(self, ip: str) -> None:
        CMD_MOTOR = "CMD_MOTOR"
        CMD_LED = "CMD_LED"
        CMD_LED_MOD = "CMD_LED_MOD"
        CMD_SERVO = "CMD_SERVO"
        CMD_BUZZER = "CMD_BUZZER"
        CMD_LIGTH = "CMD_LIGTH"
        CMD_POWER = "CMD_POWER"
        CMD_LED_TYPE = "CMD_LED_TYPE"
        CMD_START = "CMD_START"
        CMD_STOP = "CMD_STOP"
        CMD_MODE = "CMD_MODE"
        self.ip = ip
        self.servoH = 90
        self.servoV = 90
        self.motor = 0
        self.sonic = 0
        self.lightleft = 0
        self.lightright = 0
        self.power = 0
        self.neuralactive = False
        try:
            self.device = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.videodevice = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.device.connect((self.ip, 5000))
            self.videodevice.connect((self.ip, 8000))
            self.videoconnection = self.videodevice.makefile("rb")
        except Exception as e:
            print("Connecting to ", self.ip, "\n Exception: ", e)
            sys.exit()
        self.image: Image = None
        self.streamthread = Thread(target=self.streamvideo)
        self.streamthread.start()
        self.receivethread = Thread(target=self.receive)
        self.receivethread.start()

    def disconnect(self):
        self.device.shutdown(socket.SHUT_RDWR)
        self.device.close()
        self.videodevice.shutdown(socket.SHUT_RDWR)
        self.videodevice.close()
        sys.exit()

    def streamvideo(self):
        print("Start streaming video")
        while True:
            try:
                data = self.videoconnection.read(4)
                length = struct.unpack("<L", data)[:4]
                jpg = self.videoconnection.read(length[0])
                if (jpg[6:10] in (b"JFIF", b"Exif")) and jpg.rstrip(b"\0\r\n").endswith(
                    b"\xff\xd9"
                ):
                    try:
                        Image.open(io.BytesIO(jpg)).verify()
                        self.image = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                    except Exception as e:
                        print(e)
                        pass
            except Exception as e:
                print(e)
                break

    def receive(self):
        print("Start receiving data")
        data = ""
        while True:
            data += str(self.device.recv(1024), "utf-8")
            if data != "":
                print("Received: " + data)
                cmds = data.split("\n")
                if cmds[-1] != "":
                    data = cmds[-1]
                    cmds = cmds[:-1]
                for cmd in cmds:
                    message = cmd.split("#")
                    if self.CMD_SONIC in message:
                        self.sonic = message[1]
                        print("Sonic: " + self.sonic)
                    elif self.CMD_LIGHT in message:
                        self.lightleft = message[1]
                        self.lightright = message[2]
                        print("Light: " + self.lightleft + " " + self.lightright)
                    elif self.CMD_POWER in message:
                        self.power = int((float(message[1]) - 7) / 1.40 * 100)
                        print("Power: " + str(self.power))

    def look(self):
        if not self.neuralactive:
            print("Starting neural network")
            self.cudadevice = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.transform = get_transform(image_size=384)
            delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]
            model = ram(
                pretrained="./pretrained/ram_swin_large_14m.pth",
                image_size=384,
                vit="swin_l",
            )
            model.eval()
            model.to(self.cudadevice)
            self.model = model
            self.neuralactive = True
            print("Neural network started")
        print("Analyzing image")
        image = (
            self.transform(Image.fromarray(self.image)).unsqueeze(0).to(self.cudadevice)
        )
        res = inference(image, self.model)
        return res[0]
