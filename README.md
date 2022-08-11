# auroraRecognition_FMI
Repository for the files that are necessary for the auroraRecognition project

The repository includes codes that are used in Crystal that make the aurora recognition routine/method run.
The functionality of the method is made totally in Crystal. Astra has only a web page that shows the data
that has been pushed from Crystal to Astra.

The getLatest -codes for both KEV and MUO handle the functionality of the routine/method, i.e.
get the image from an ASC video, crop the trees etc., classifies the image as aurora/no-aurora and
push the image to Astra. Please refer to the doumentation paper for more specific information.

The morningRoutine -codes for KEV and MUO handle the task of compressing the data and "cleaning" the 
folders in Crystal after the day/night when the ASC camera does not take pictures any more. The classified
images are compressed to a tar-package and the filenames, i.e. the timestamps of the images, are collected
to a list. These are then pushed to Astra by this routine.

There are also a couple of shell scripts in the repository. Their function is to actually "run" the above
mentioned codes. They activate the conda environment and the virtual environment created for Python in the 
server. Finally, they call the Python codes and make them run. The shell scripts are the called from crontab
in Crystal. Otherwise, a user would have to manually run the command "python3 getLatest...etc.". The crontab
that I've been using in my personal Crystal folder can also be found from the repository and also the instructions
for creating the conda environment in Crystal and the package requirements in order for the codes to work.


