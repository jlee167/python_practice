import serial
from PIL import Image, ImageShow
import numpy as np
from io import BytesIO
import cv2
import time
import wave

import matplotlib.pyplot as plt
from pydub import AudioSegment

uart_handle = serial.Serial('COM8', 921600)
frame = [0]
#uart_handle.close()


def capture_frame():

    # Open Serial Port
    #uart_handle.open()

    # 0:3 are image length (32 bit int), 4:5 are JPEG (0xff, 0xd8)
    jpeg_header = [0, 0, 0, 0, 0, 0]
    print("Init: " + str(time.time()))
    # Find JPEG SOF header 0xff, 0xd8 in UART stream
    if frame[0] == 0:
        while (jpeg_header[4] != b'\xff') or (jpeg_header[5] != b'\xd8'):
            if uart_handle.inWaiting():
                jpeg_header.pop(0)
                jpeg_header.append(uart_handle.read(1))
    else:
        for i in range(0, 6):
            jpeg_header[i] = uart_handle.read(1)
            print("loop" + str(i) + " : " + str(time.time()))
    print("Find hEADER: " + str(time.time()))

    frame[0] = 1

    # Get JPEG file length in bytes from 4 bytes before JPEG SOF header
    bytelist = bytearray(4)
    for i in range(0, 4):
        bytelist[i] = int.from_bytes(jpeg_header[i], byteorder='little')
    num_bytes = int.from_bytes(bytelist, byteorder='little')
    #print(num_bytes)

    # Receive rest of image and close UART port
    while not uart_handle.inWaiting():
        pass
    recv_bytes = uart_handle.read(num_bytes - 2)
    #uart_handle.close()
    print("Get Uart: " + str(time.time()))

    # Write above image to jpeg file

    image = open('capture2.jpeg', 'wb')
    image.write(b'\xff')
    image.write(b'\xd8')
    image.write(recv_bytes)
    image.close()


    bitmap = np.frombuffer(b'\xff' + b'\xd8' + bytes(recv_bytes), dtype=np.uint8)
    #file_jpgdata = BytesIO(recv_bytes)
    # dt = Image.open(file_jpgdata)
    # ImageShow.show(dt, "noname")
    # r,g,b = Image.fromarray()
    print("Bitmap: " + str(time.time()))

    #uart_handle.close()
    uart_handle.flush()
    return cv2.imdecode(bitmap, cv2.IMREAD_UNCHANGED)


"""
while (True):
    print("Pre: " + str(time.time()))
    img = capture_frame()
    print("Aft: " + str(time.time()))
    cv2.imshow("title", img)
    print("Show: " + str(time.time()))
    cv2.waitKey(1)
"""


def nothing(x):
    pass


filter_img = cv2.imread("capture2.jpeg")
gray = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("Title")
cv2.createTrackbar('K','Title', 1, 30, nothing)

while(1):
    cv2.waitKey(100)
    k = cv2.getTrackbarPos('K', 'Title')
    if k == 0:
        k = 1

    kernel = np.ones((k, k), np.float32) / (k ** 2)
    """
    kernel = np.array([
        [0.1, 0.1, 0.1],
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.1]
    ])
    """
    #dst = cv2.filter2D(filter_img, -1, kernel)
    #dst = cv2.medianBlur(filter_img, k + (k%2 + 1))
    dst = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    cv2.imshow('Title', dst)
"""
sound = AudioSegment.from_wav('audio_sample.wav')
sound = sound.set_channels(1)
sound.export('out.wav', format='wav')

audio = wave.open('out.wav', 'r')

signal = audio.readframes(-1)
signal = np.fromstring(signal, dtype=np.int16)


freq_domain = np.fft.fft(signal, n=512)
print(np.size(freq_domain))

n_channels = audio.getnchannels()

#print(freq_domain)
plt.figure(1)
plt.title('Audio Sample')
plt.plot(freq_domain)
plt.show()
"""