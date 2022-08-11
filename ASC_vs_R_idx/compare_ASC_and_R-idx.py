import csv
import os
import datetime
import glob
from PIL import Image
import imageClassification

CSV_FOLDER = '/home/moisio/r-index/KEVO/test'
CSV_FILES = glob.glob(CSV_FOLDER + '/*')
VIDEO_FOLDERS = '/home/moisio/KEV/2021-2022'
TIME_COORDINATES = (500, 0, 608, 30)
DATE_COORDINATES = (0, 0, 120, 30)
RESULT_DATA_CSV = '/home/moisio/r-index/KEVO/test/test_results.csv'
TREE_MASK_PATH = '/home/moisio/Documents/auroraRecognition/mask.png'
SPEEDING_FACTOR = 0.01


result_csv, csv_writer = imageClassification.makeResultCsv(RESULT_DATA_CSV)
TREE_MASK = Image.open(TREE_MASK_PATH)
for csv_filename in CSV_FILES:
    with open(csv_filename, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:  # the basic name actions can be made into a function
            r_idx_val, r_idx_level, r_idx_t_min_5, r_idx_t_0 = imageClassification.getRidxTimestamp(row)
            videoFolder = VIDEO_FOLDERS + '/' + r_idx_t_min_5.strftime("%d%m%Y")
            if not os.path.exists(videoFolder):
                continue
            videoName, videoFolder = imageClassification.searchVideo(VIDEO_FOLDERS, videoFolder, r_idx_t_min_5)
            if videoName is None:
                continue
            # Read the videoName and try to make an educated guess of the specific frame that should be read
            videoTimestamp, timeDifference = imageClassification.getTimeDifference(videoName, r_idx_t_min_5)
            videoReader, totalFrames = imageClassification.readVideo('/'.join([videoFolder, videoName]))
            if timeDifference > totalFrames:
                continue
            startFrame, framesToBeRead, videoReader = imageClassification.setStartFrame(timeDifference, totalFrames,
                                                                                        videoReader, SPEEDING_FACTOR)
            # Save the frame and check the time that it's roughly in the time slot
            videoReader, firstFrameTimestamp, firstFrame, frameCount, startFrame = imageClassification.readFirstFrame(
                videoReader, TIME_COORDINATES, DATE_COORDINATES, framesToBeRead, startFrame)

            lowerBoundary = r_idx_t_min_5 - datetime.timedelta(minutes=1)
            upperBoundary = r_idx_t_min_5 + datetime.timedelta(minutes=1)
            if not lowerBoundary <= firstFrameTimestamp and firstFrameTimestamp <= upperBoundary:
                continue
            RidxStatRow = [r_idx_t_0.strftime("%Y-%m-%d %H:%M:%S+00:00"), r_idx_val, r_idx_level]
            ASCStatRow = imageClassification.getASCStats(firstFrame, TREE_MASK, videoReader, frameCount,
                                                         firstFrameTimestamp)
            combinedStatRow = RidxStatRow + ASCStatRow
            csv_writer.writerow(combinedStatRow)
            videoReader.release()

csv_file.close()
TREE_MASK.close()
