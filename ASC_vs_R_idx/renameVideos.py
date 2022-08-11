# 1: Hae oikea kansio ja kaikki kansiot siellä
# 2: Looppaa kansiot läpi ja videot niiden sisällä
# 3: Muuta nimi muotoon yyyy-mm-dd hh:mm utc, tunnista kellonaika tesseractilla.
# 3: Pakota tesseract tunnistamaan numeerinen aika. Ota seuraava freimi, jos ei onnistu
import glob
import os
import cv2
from PIL import Image
from pytesseract import image_to_string


def getTimeStamp(image, timeCoordinates):
    timeImage = image.crop(timeCoordinates).resize((250, 60)).convert('L')
    time = image_to_string(timeImage).rstrip().replace('-', ':')[:-2]
    time = ''.join(filter(str.isalnum, time))
    return time

SRC_FOLDER = '/home/moisio/KEV/2021-2022'
FOLDERS = sorted(os.listdir(SRC_FOLDER))
TIME_COORDINATES = (500, 0, 608, 30)

for folder in FOLDERS:
    VIDEOS = sorted(glob.glob(SRC_FOLDER + '/' + folder + '/*.avi'))

    for video in VIDEOS:
        videoCapture = cv2.VideoCapture(video)
        success, frame = videoCapture.read()
        while not success:
            success, frame = videoCapture.read()

        image = Image.fromarray(frame)
        time = str(getTimeStamp(image, TIME_COORDINATES))
        while not time.isnumeric():
            success, frame = videoCapture.read()
            image = Image.fromarray(frame)
            time = str(getTimeStamp(image, TIME_COORDINATES))

        time = ':'.join([time[:2], time[2:4]])
        oldName = video.split('/')[-1]
        videoPath = '/'.join(video.split('/')[:-1])
        yyyy_mm_dd = [oldName[4:8], oldName[2:4], oldName[:2]]
        newName = ' '.join(['-'.join(yyyy_mm_dd), time, 'utc']) + '.avi'
        newPath = '/'.join([''.join(videoPath), newName])

        print(newPath)
        os.rename(video, newPath)




