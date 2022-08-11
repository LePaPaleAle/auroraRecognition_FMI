import datetime
import os
from glob import glob
import tarfile
from subprocess import run


''' This function is used for creating a tar package from a given directory.'''


def createTarFile(outputName, sourceDir):
    with tarfile.open(outputName, 'w:gz') as tar:
        tar.add(sourceDir, arcname=os.path.basename((sourceDir)))


''' cleanDirectory is used to clean a directory from files with the given ending.
    Mainly used for deleting png-files as the loop suggests.'''


def cleanDirectory(directoryPath, ending):
    for pngFile in glob('/*'.join([directoryPath, ending])):
        os.remove(pngFile)


''' checkDir is just a function that is used to check if a certain directory exists.
    If not, the directory is created.'''


def checkDir(dirPath):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)


MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']

CRYSTAL_AURORA_PATH = '/home/users/moisio/testing/MUO/aurora'
CRYSTAL_NO_AURORA_PATH = '/home/users/moisio/testing/MUO/no-aurora'
CRYSTAL_STATS_PATH = '/home/users/moisio/testing/MUO/stats'

ASTRA_AURORA_PATH = 'moisio@astra.fmi.fi:/home/users/moisio/public_html/testing/MUO/aurora'
ASTRA_NO_AURORA_PATH = 'moisio@astra.fmi.fi:/home/users/moisio/public_html/testing/MUO/no-aurora'
ASTRA_STATS_PATH = 'moisio@astra.fmi.fi:/home/users/moisio/public_html/testing/MUO/stats/'

# Get the current month in order to direct the timestamps to correct lists
current_day = datetime.datetime.today()
current_month = MONTHS[current_day.month - 1]

# Create the image timestamp list txt-files for the ongoing month if not yet created
# These are the lists that are made in order to archive the monthly lists
aurora_stats_file_name = CRYSTAL_STATS_PATH + '/aurora/aurora_list_MUO_' + current_month + '.txt'
aurora_stats_file_name_astra = aurora_stats_file_name.split('/')[-1]
no_aurora_stats_file_name = CRYSTAL_STATS_PATH + '/no-aurora/no-aurora_list_MUO_' + current_month + '.txt'
no_aurora_stats_file_name_astra = no_aurora_stats_file_name.split('/')[-1]
if not os.path.exists(aurora_stats_file_name):
    auroraFile = open(aurora_stats_file_name, 'w+')
    auroraFile.write('Images with aurora in Muonio during {}:\n'.format(current_month))
    auroraFile.write('Format: MUONIO_YYYYMMDD_hh:mm:ss\n\n')
    auroraFile.close()

if not os.path.exists(no_aurora_stats_file_name):
    noAuroraFile = open(no_aurora_stats_file_name, 'w+')
    noAuroraFile.write('Images with no aurora in Muonio during {}:\n'.format(current_month))
    noAuroraFile.write('Format: MUONIO_YYYYMMDD_hh:mm:ss\n\n')
    noAuroraFile.close()

# Read the contents of the aurora and no-aurora folders
aurora_images = [os.path.basename(fileName) + '\n' for fileName in
                 sorted(glob(CRYSTAL_AURORA_PATH + '/*.jpg'))]
no_aurora_images = [os.path.basename(fileName) + '\n' for fileName in
                    sorted(glob(CRYSTAL_NO_AURORA_PATH + '/*.jpg'))]

with open(aurora_stats_file_name, 'a+') as aurora_list_file:
    aurora_list_file.writelines(aurora_images)
with open(no_aurora_stats_file_name, 'a+') as no_aurora_list_file:
    no_aurora_list_file.writelines(no_aurora_images)

# Write the filenames, i.e. timestamps, to these current month lists
# These lists are the ones that are being displayed on the web page in Astra
current_aurora_stats_file_name = CRYSTAL_STATS_PATH + '/current_month_aurora_images_MUO.txt'
current_aurora_stats_file_name_astra = current_aurora_stats_file_name.split('/')[-1]
current_no_aurora_stats_file_name = CRYSTAL_STATS_PATH + '/current_month_no_aurora_images_MUO.txt'
current_no_aurora_stats_file_name_astra = current_no_aurora_stats_file_name.split('/')[-1]

# Write the "current" stat files that shall be seen at the website
with open(current_aurora_stats_file_name, 'w+') as current_aurora_stats_file:
    with open(aurora_stats_file_name, 'r') as current_aurora_stats:
        aurora_stat_lines = current_aurora_stats.readlines()
        current_aurora_stats_file.writelines(aurora_stat_lines)
with open(current_no_aurora_stats_file_name, 'w+') as current_no_aurora_stats_file:
    with open(no_aurora_stats_file_name, 'r') as current_no_aurora_stats:
        no_aurora_stat_lines = current_no_aurora_stats.readlines()
        current_no_aurora_stats_file.writelines(no_aurora_stat_lines)

# Create the tar packages from images in the aurora and no-aurora folders
auroraTarFileName = CRYSTAL_AURORA_PATH + '/aurora_MUO.tar.gz'
noAuroraTarFileName = CRYSTAL_NO_AURORA_PATH + '/no-aurora_MUO.tar.gz'

# Check that the directories still exist
checkDir(CRYSTAL_AURORA_PATH)
checkDir(CRYSTAL_NO_AURORA_PATH)

# Create the tar files from the directories
createTarFile(auroraTarFileName, CRYSTAL_AURORA_PATH)
createTarFile(noAuroraTarFileName, CRYSTAL_NO_AURORA_PATH)

# Clean the image directories
cleanDirectory(CRYSTAL_AURORA_PATH, '.jpg')
cleanDirectory(CRYSTAL_NO_AURORA_PATH, '.jpg')

# push the tar-packages to Astra
run(['rsync', '--timeout=30', '-avP', auroraTarFileName,
     ASTRA_AURORA_PATH])
run(['rsync', '--timeout=30', '-avP', noAuroraTarFileName,
     ASTRA_NO_AURORA_PATH])

# Push the current month timestamp lists to Astra for display on the web page
run(['rsync', '--timeout=30', '-avP', current_aurora_stats_file_name,
     ASTRA_STATS_PATH + current_aurora_stats_file_name_astra])
run(['rsync', '--timeout=30', '-avP', current_no_aurora_stats_file_name,
     ASTRA_STATS_PATH + current_no_aurora_stats_file_name_astra])

# Push the "copies" of the current month lists in order to archive them in Astra
run(['rsync', '--timeout=30', '-avP', aurora_stats_file_name,
     ASTRA_STATS_PATH + '/aurora/' + aurora_stats_file_name_astra])
run(['rsync', '--timeout=30', '-avP', no_aurora_stats_file_name,
     ASTRA_STATS_PATH + '/no-aurora/' + no_aurora_stats_file_name_astra])


