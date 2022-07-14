# PhD-project

For more details on NOW corpus & HPC usage, please refer to [this documentation](https://docs.google.com/document/d/1pJ6w0NcR076oyPCq2k2JkogblUKxICGdEs_JHLW4bsA/edit?usp=sharing).

## Data 
The following three files can be accessd via [`/RelMod/model_input/`](https://webdata.tudelft.nl/staff-umbrella/RelMod/model_input/) in TU Delft Webdata, unless otherwise stated.  
- `large_model_input_gs1.pkl` - 21,98 GB. It can be downloaded [here](https://drive.google.com/file/d/1NVzG8RwEKiyk-dMIFUOicELLEQhMAFh8/view). This link is from LARN paper. 
- `larn_data_smaller.pt` - 11,45 GB. The same contents as `large_model_input_gs.pkl` but the data type is 32-bit. You can chhose to use this one or the previous one (which might depend on how much memory you have). 
- `target_word_ix_counter.pk` - 234KB. Counts of each unique word in the corpus. 
- `LARN-us-ru-4tp.pk` - 

The following two files are included in this github repository:
- `embeddings*` - topic embeddings. They are included in this repository under the folder `/ARM/Topic/`.   
- `trained-model*` - outputs from ARM model.  


## Command Line Arguments 

To train the model, you can run the following command line. To run the model on HPC, please use the sbatch files `ARM.sbatch` in the folder `ARM/sbatch/`.
```
python ARM.py
```

To produce the temporal trend plot, you can run the following command line. To run the model on HPC, please use the sbatch files `tmp-trend.sbatch` in the folder `ARM/sbatch/`.
```
python TemporalTrend.py
```


## Files and Directories 
- `ARM/` 
  - `Topic/` 
    - `info*` - outputs from [BERTopic](https://maartengr.github.io/BERTopic/api/bertopic.html). There are 10 clusters of topics per pair of country, each cluster has 10 keywords. This file contains the [c-TF-IDF score](https://maartengr.github.io/BERTopic/api/ctfidf.html) of each keywords. 
    - `prob*` - outputs from BERTopic. They are the probability of the assigned topic per document.
    - `embeddings*` -  topic embeddings by multiplying document embeddings (generated by [sentence transformer](https://www.sbert.net/examples/applications/computing-embeddings/README.html) with pretrained model *all-mpnet-base-v2*) with `prob*`.  
  - `sbatch/` - the sbatch files that can be submitted to HPC. Please refer to the documentation if you need more info of HPC. 
  - `plot` - temporal plots per topic that are created by `TemporalTrend.py`
  - `TopicEmbedding.ipynb` - the script used to create `info*`, `prob*` and `embeddings*`. 
  - `ARM.py` - trains the model.    
  - `TemporalTrend.py` - produces temporal trend per topic. The output plots are in `plot/` in this github repository.
  - `constants.py` - indicates the constants that are used in other scripts. You can change the data directories here.
  - `modules.py` - contains modules of ARM. 
  - `preprocessing.py`- is supposed to preprocess the raw textual data from May 2019 to April 2022 (which can be downloaded on TU Delft Webdata.) But I have not yet tried this script yet. This scrips is also from LARN paper. 
  - `utils.py` - contains some common functions. 
- `LARN-all-data/` contains the LARN model which can be run with all data (i.e., the original model from the paper.) 
- `LARN/`contains the LARN model which can be run with topic-related articles. 
  - `FilterArticle.py` - selects the articles that contains at least one keywords from each topic.  
  

## ARM model structure
