# ArchaicToModernITA_MNLP

##How to run all the code
All the code in this repo are meant to be run on Cineca, so first of all let's se how to download locally a model from HuggingFace

###Download the model
First of all you'll need to enter the Cineca environment, and from the $HOME area mode to the $SCRATH area. Here create a folder, this will be the folder were all the model will be downloaded, and get here absolute path, you can do this be entering the folder and type:
```
pwd
```
Now go back to the $HOME area and create a virtual environment, enter it and download Python and all the libraries in the "requirment.txt" file. After that run:
```
export HF_HOME=<path/to/the/folder/in/scratch>
```
to set that folder as the one where HuggingFace will download every model. After that you can run the following command to login and dowload the model from HuggingFace:
```
huggingface-cli login
huggingface-cli download <name_of_the_model_to_download>
```
