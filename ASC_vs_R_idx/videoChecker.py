import csv
import os
import datetime
import cv2
import pytesseract
from PIL import Image
import numpy as np
import glob


def getFrameTimeStamp(image, time_coordinates, date_coordinates):
    timeImage = image.crop(time_coordinates)
    dateImage = image.crop(date_coordinates)
    timeImage = timeImage.resize((250, 60)).convert('L')
    dateImage = dateImage.resize((250, 60)).convert('L')
    time = str(pytesseract.pytesseract.image_to_string(timeImage).rstrip())
    date = pytesseract.pytesseract.image_to_string(dateImage).rstrip()
    time = ''.join(filter(str.isnumeric, time))
    date = ''.join(filter(str.isnumeric, date))
    if len(date) == 6 and len(time) == 6:
        try:
            timestamp = datetime.datetime(year=2000 + int(date[4:6]), month=int(date[2:4]), day=int(date[:2]),
                                          hour=int(time[:2]), minute=int(time[2:4]), second=int(time[4:6]))
        except ValueError or TypeError:
            timestamp = None
            print('Error in forming timestamp!\ndate: {}\ntime: {}'.format(date, time))
    else:
        timestamp = None
    return timestamp


def checkVideo(videoPath, videoCapObject, timeCoordinates, dateCoordinates):
    old_path = videoPath.split('/')[:-1]
    videoCapObject.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, firstFrame = videoCapObject.read()
    videoCapObject.set(cv2.CAP_PROP_POS_FRAMES, 2)
    ret2, secondFrame = videoCapObject.read()
    try:
        firstImage = Image.fromarray(firstFrame)
        secondImage = Image.fromarray(secondFrame)
        timestamp1 = getFrameTimeStamp(firstImage, timeCoordinates, dateCoordinates)
        timestamp2 = getFrameTimeStamp(secondImage, timeCoordinates, dateCoordinates)
    except AttributeError or ValueError or TypeError:
        timestamp1, timestamp2 = None, None

    if timestamp1 is None or timestamp2 is None:
        print('Cannot check {}'.format(videoPath))
    else:
        time_difference = (timestamp2 - timestamp1).total_seconds()
        if time_difference > 2:
            new_name = timestamp2.strftime("%d%m%Y_%H%M%S.avi")
            new_path = '/' + '/'.join(old_path) + '/' + new_name
            writeVideo(videoCapObject, new_path)
            os.remove(videoPath)


def writeVideo(videoReader, videoPath):
    videoReader.set(cv2.CAP_PROP_POS_FRAMES, 2)
    fps = videoReader.get(cv2.CAP_PROP_FPS)
    four_cc = cv2.VideoWriter_fourcc(*'mp4v')
    # four_cc = int(videoReader.get(cv2.CAP_PROP_FOURCC))
    frame_size = (int(videoReader.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(videoReader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(videoPath, four_cc, fps, frame_size)

    ret, frame = videoReader.read()
    while ret:
        videoWriter.write(frame)
        ret, frame = videoReader.read()

    videoReader.release()
    videoWriter.release()


VIDEO_FOLDER = '/home/moisio/KEV/2021-2022 testing copy'
TIME_COORDINATES = (500, 0, 608, 30)
DATE_COORDINATES = (0, 0, 120, 30)

for folder in sorted(os.listdir(VIDEO_FOLDER)):
    for video in sorted(glob.glob(VIDEO_FOLDER + '/' + folder + '/*.avi')):
        videoCapturer = cv2.VideoCapture(video)
        checkVideo(video, videoCapturer, TIME_COORDINATES, DATE_COORDINATES)




