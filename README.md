# ArchaicToModernITA_MNLP

## How to run all the code
All the code in this repo are meant to be run on Cineca, so first of all let's se how to download locally a model from HuggingFace

### Download the model
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
You can now download all the model that we used for this project: [Zephyr_7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), [Llama_3.2_3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [NLLB_200_1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B), [Prometheus_7B](https://huggingface.co/Unbabel/M-Prometheus-7B).
After that go back to the folder in $SCRATCH, here you should find all the model folder, open it and get the path up to the snapshot, something like this:
```
"/leonardo_scratch/large/userexternal/<your_username>/hf_cache/hub/models--Unbabel--M-Prometheus-7B/snapshots/030fb74806e4228c466a98706a297d43b31ce5df"
```
and put this path in the .py file in the variable "model_id" or "model_path". Eventually you need to replace the following path in the .slurm.sh file:
```
export HF_HOME=/leonardo_scratch/large/userexternal/<your_username>/hf_cache/hub
export HF_DATASETS_CACHE=/leonardo_scratch/large/userexternal/<your_username>/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=/leonardo_scratch/large/userexternal/<your_username>/hf_cache/hub
```
with the èath tp the folder in $SCRATCH.

### Zephyr Translation
