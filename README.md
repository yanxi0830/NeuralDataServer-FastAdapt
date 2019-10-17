# NeuralDataServer-FastAdapt
This repo contains standalone scripts for the fast adaptation module that can be used off-the-shelf to generate transfer performance. 

Running the scripts assumes a directory `[PATH/TO/CLIENT/DATASET]` containing images of the client dataset, 
and a directory `[PATH/TO/DOWNLOADED/EXPERT/MODELS]` containing pre-trained weights of expert models. 

```bash
conda create --name fast-adapt
conda install pytorch torchvision
conda install torchnet
```

## Usage
```bash
cd src
python main.py --imagedir [PATH/TO/CLIENT/DATASET] --experts_dir [PATH/TO/DOWNLOADED/EXPERT/MODELS]
```

Running it off-the-shelf will generate a `z.pickle` file inside `experiments/GenericFastAdapt/` containing the performance metric for each expert. 
Send this file back to the server to obtain a subset of server data that is most relevant for images inside `[PATH/TO/CLIENT/DATASET]`. 
