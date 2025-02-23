# Finetune-llama
Simple fine tune process to Ollama model, using data from gitlab repository.

It is necessary NVIDIA GPU! (Or use on Colab Cloud Notebook).

The new model will be generated in the model folder with the name unsloth.Q8_0.gguf.

A file called "Modelfile" will also be generated in this folder.

To use the model on Ollama to execute the following steps:

- ollama create name_model -f Modelfile

To check:

- ollama list

The script gitlab_get.py is just an example to generate data for you to use. If you prefer, create a JSON file with your data, following the format of the "data_sample.jsonl" file.

