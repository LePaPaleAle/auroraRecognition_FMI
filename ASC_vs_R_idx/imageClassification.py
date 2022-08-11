import csv
import os
import datetime
import cv2
import pytesseract
from PIL import Image
import numpy as np


def getTimeStamp(image, time_coordinates, date_coordninates):
    timeImage = image.crop(time_coordinates)
    dateImage = image.crop(date_coordninates)
    timeImage = timeImage.resize((250, 60)).convert('L')
    dateImage = dateImage.resize((250, 60)).convert('L')
    time = str(pytesseract.pytesseract.image_to_string(timeImage).rstrip())
    date = pytesseract.pytesseract.image_to_string(dateImage).rstrip()
    time = ''.join(filter(str.isnumeric, time))
    date = ''.join(filter(str.isnumeric, date))

    if len(time) == 6 and len(date) == 6:
        try:
            hours, minutes, seconds = int(time[:2]), int(time[2:4]), int(time[4:6])
            day, month, year = int(date[:2]), int(date[2:4]), 2000 + int(date[4:6])
            firstFrameTimestamp = datetime.datetime(year=year, month=month, day=day, hour=hours, minute=minutes,
                                                    second=seconds)
            return firstFrameTimestamp
        except ValueError or TypeError:
            return None
    else:
        return None




def readFirstFrame(videoCapObject, timeCoordinates, dateCoordinates, framesToBeRead, startFrame):
    success, firstFrame = videoCapObject.read()
    framesToBeRead -= 1
    startFrame += 1
    while not success:
        success, firstFrame = videoCapObject.read()
        framesToBeRead -= 1
        startFrame += 1
    image = Image.fromarray(firstFrame)
    addedSeconds = 0
    firstFrameTimeStamp = getTimeStamp(image, timeCoordinates, dateCoordinates)
    while firstFrameTimeStamp is None:
        ret, frame = videoCapObject.read()
        image = Image.fromarray(frame)
        firstFrameTimeStamp = getTimeStamp(image, timeCoordinates, dateCoordinates)
        addedSeconds += 1

    firstFrameTimestamp = firstFrameTimeStamp - datetime.timedelta(seconds=addedSeconds)

    return videoCapObject, firstFrameTimestamp, firstFrame, framesToBeRead, startFrame



def makeResultCsv(filePath):
    if os.path.exists(filePath):
        result_csv_file = open(filePath, 'a+')
        csvWriter = csv.writer(result_csv_file)
    else:
        result_csv_file = open(filePath, 'w+')
        csvWriter = csv.writer(result_csv_file)
        firstRow = ['Ri_date', 'Ri_val', 'Ris_en', 'ASC_max_date', 'ASC_Class', 'Aurora/Cloud-%', 'no class-%']
        csvWriter.writerow(firstRow)

    return result_csv_file, csvWriter


def cropTrees(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    mask = mask == 255
    image[mask] = (0, 0, 0)

    ind = np.where(~mask)

    min_y = np.amin(ind[0]) - 1
    max_y = np.amax(ind[0]) + 2
    min_x = np.amin(ind[1]) - 1
    max_x = np.amax(ind[1]) + 2

    image = image[min_y:max_y, min_x:max_x, :]
    mask = mask[min_y:max_y, min_x:max_x]

    return image, mask


def getASCStats(firstFrame, treeMask, vidReader, framesToBeRead, firstFrameTimeStamp):
    labels, classPercentages, noClassPercentages, timestamps = [], [], [], []
    imageLabel, classCoverage, noClassCoverage = classifyImage(firstFrame, treeMask)
    labels.append(imageLabel), classPercentages.append(classCoverage), noClassPercentages.append(noClassCoverage)
    timestamps.append(firstFrameTimeStamp)

    for i in range(int(framesToBeRead)):
        success, frame = vidReader.read()
        if not success:
            continue
        imageLabel, classCoverage, noClassCoverage = classifyImage(frame, treeMask)
        frameTimeStamp = firstFrameTimeStamp + datetime.timedelta(seconds=int(i))
        timestampString = frameTimeStamp.strftime("%Y-%m-%d %H:%M:%S+00:00")
        labels.append(imageLabel), classPercentages.append(classCoverage), noClassPercentages.append(noClassCoverage)
        timestamps.append(timestampString)

    if 'Aurora' in labels:
        label = 'Aurora'
    else:
        label = 'No Aurora'
    maxClassPercentage = max(classPercentages)
    maxIndex = classPercentages.index(maxClassPercentage)
    maxNoClassPercentage = noClassPercentages[maxIndex]
    maxTimestamp = timestamps[maxIndex]
    statRow = [maxTimestamp, label, maxClassPercentage, maxNoClassPercentage]

    return statRow


def getRidxTimestamp(r_idx_row):
    date, time = r_idx_row[0].split(' ')[0], r_idx_row[0].split(' ')[1][:5]
    R_idx_value = r_idx_row[2]
    R_idx_level = r_idx_row[4]
    # Manipulate the date to the correct format ddmmyyyy
    dd, mm, yyyy = int(date.split('-')[2]), int(date.split('-')[1]), int(date.split('-')[0])
    hour, minute = int(time[:2]), int(time[3:5])
    timestamp_t_5 = datetime.datetime(year=yyyy, month=mm, day=dd, hour=hour, minute=minute) - datetime.\
        timedelta(minutes=5)
    timestamp_t_0 = datetime.datetime(year=yyyy, month=mm, day=dd, hour=hour, minute=minute)

    return R_idx_value, R_idx_level, timestamp_t_5, timestamp_t_0


def getYesterdayLastVideo(vidFolder, date):
    dd, mm, yyyy = int(date[:2]), int(date[2:4]), int(date[4:])
    thisday = datetime.datetime(year=yyyy, month=mm, day=dd)
    yesterday = thisday - datetime.timedelta(days=1)
    formatted = yesterday.strftime('%d%m%Y')

    videoFolder = '/'.join([vidFolder, formatted])
    video = None
    if os.path.exists(videoFolder):
        videos = sorted(os.listdir(videoFolder))
        videos.reverse()
        hour = 23
        for videoFile in videos:
            if hour == int(videoFile[9:11]):
                video = videoFile
                break

    return video, videoFolder



def searchVideo(videoFolders, videoFolder, timestamp):
    videos = sorted(os.listdir(videoFolder))
    video = None
    for idx, vidName in enumerate(videos):
        vidHour, vidMinute = int(vidName[9:11]), int(vidName[11:13])
        if timestamp.hour == 0 and timestamp.minute < vidMinute:
            video, videoFolder = getYesterdayLastVideo(videoFolders, timestamp.strftime("%d%m%Y"))
            break
        elif timestamp.hour == vidHour and timestamp.minute >= vidMinute:
            video = vidName
            break
        elif timestamp.hour == vidHour and timestamp.minute < vidMinute and \
                timestamp.hour - 1 == int(videos[idx - 1][9:11]):
            video = videos[idx - 1]
            break

    return video, videoFolder


def getTimeDifference(videoName, RidxTimestamp):
    videoDay, videoMonth, videoYear, videoHour, videoMinute, videoSecond = int(videoName[:2]), int(videoName[2:4]), \
                                                                           int(videoName[4:8]), int(
        videoName[9:11]), int(videoName[11:13]), int(videoName[13:15])
    # calculate the time difference
    videoTimestamp = datetime.datetime(year=videoYear, month=videoMonth, day=videoDay,
                                       hour=videoHour, minute=videoMinute, second=videoSecond)
    timeDifference = (RidxTimestamp - videoTimestamp).total_seconds()

    return videoTimestamp, timeDifference


def readVideo(vidPath):
    videoReader = cv2.VideoCapture(vidPath)
    frameCount = videoReader.get(cv2.CAP_PROP_FRAME_COUNT)

    return videoReader, frameCount


def setStartFrame(timeDifference, numFrames, videoReader, speeding_factor):
    timeDifference = min(timeDifference, numFrames - 300)
    skippedSeconds = round(speeding_factor * timeDifference)
    startFrame = timeDifference - skippedSeconds
    framesToBeRead = 300
    if timeDifference < 0:
        startFrame = 0
        framesToBeRead = 300 - abs(timeDifference - skippedSeconds)

    videoReader.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    return startFrame, framesToBeRead, videoReader


def classifyImage(frame, treeMask):
    image = Image.fromarray(frame)
    image, mask = cropTrees(image, treeMask)
    height, width, _ = np.shape(image)
    classifiedImage = np.full_like(np.zeros((height, width)), False, dtype=bool)
    R, G, B = image[:, :, 0] / 255., image[:, :, 1] / 255., image[:, :, 2] / 255.
    R = image[:, :, 0]
    N = np.size(R[np.nonzero(R)])
    minimum = np.minimum(np.minimum(R, G), B)
    maximum = np.maximum(np.maximum(R, G), B)
    L = (minimum + maximum) / 2

    green_mask = (R <= G) & (R >= B) & (G >= B) & (L > 0.4) & (L <= 0.98) & \
                 ((R - G) < -0.01) & ((R - B) > 0.01) & ((G - B) > 0.01)
    cyan_mask = (R <= G) & (R <= B) & (G >= B) & (L > 0.4) & (L <= 0.98) & \
                ((R - G) < -0.01) & ((R - B) < -0.01) & ((G - B) > 0.01)
    red_mask_1 = (R >= G) & (R >= B) & (G >= B) & (L >= 0.45) & (L <= 0.85) & \
                 ((R - G) > 0.01) & (R - B > 0.18) & ((G - B) > 0.018)
    red_mask_2 = (R >= G) & (R >= B) & (G <= B) & (L >= 0.45) & (L <= 0.85) & \
                 ((R - B) > 0.015)
    purple_mask = (R >= G) & (R <= B) & (G <= B) & (L > 0.5) & (L <= 0.98) & \
                  ((R - G) > 0.03) & ((R - B) < -0.12) & ((G - B) < -0.15)
    cloud_mask = (L < 0.85) & \
                 (((-0.01 <= (R - G)) & ((R - G) <= 0.01)) |
                 ((-0.01 <= (R - B)) & ((R - B) <= 0.01)) |
                 ((-0.01 <= (G - B)) & ((G - B) <= 0.01)))

    classifiedImage[green_mask] = 1
    classifiedImage[red_mask_1] = 1
    classifiedImage[red_mask_2] = 1
    classifiedImage[purple_mask] = 1
    classifiedImage[cyan_mask] = 1
    classifiedImage[cloud_mask] = 2
    classificationPixels = classifiedImage[np.invert(mask)]
    auroraPixelCount = classificationPixels[classificationPixels == 1].size
    cloudPixelCount = classificationPixels[classificationPixels == 2].size
    noAuroraPixelCount = classificationPixels.size - auroraPixelCount
    noCloudPixelCount = classificationPixels.size - cloudPixelCount
    auroraCoverage = 100 * round(auroraPixelCount / N, 3)
    noAuroraCoverage = 100 * round(noAuroraPixelCount / N, 3)
    cloudCoverage = 100 * round(cloudPixelCount / N, 3)
    noCloudCoverage = 100 * round(noCloudPixelCount / N, 3)
    if auroraPixelCount >= 2500:
        imageLabel = 'Aurora'
        return imageLabel, auroraCoverage, noAuroraCoverage
    else:
        imageLabel = 'No-Aurora'
        return imageLabel, cloudCoverage, noCloudCoverage
    