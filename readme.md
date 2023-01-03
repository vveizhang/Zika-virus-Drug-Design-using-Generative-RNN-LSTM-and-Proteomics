<p align="center">
<br>
</p>

# Zika virus Drug Design using Generative RNN-LSTM and Proteomics

## Table of Contents

- [Zika virus Drug Design using Generative RNN-LSTM and Proteomics](#zika-virus-drug-design-using-generative-rnn-lstm-and-proteomics)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Zika Virus](#11-zika-virus)
    - [1.2 Proteomics](#12-proteomics)
  - [1.3 Deep Learning in Drug Discovery](#13-deep-learning-in-drug-discovery)
  - [2. Pipeline](#2-pipeline)
  - [3. Differentially expressed proteins](#3-differentially-expressed-proteins)
  - [4. Training and generation phase of LSTM](#4-training-and-generation-phase-of-lstm)
    - [4.1. Create an Amazon SageMaker notebook instance](#41-create-an-amazon-sagemaker-notebook-instance)
    - [4.2. Training and Generation](#42-training-and-generation)
      - [4.2.1 Load the pre-trained Bert model and tokenizer](#421-load-the-pre-trained-bert-model-and-tokenizer)
      - [4.2.2 Model Fine-Tuning using validated durgs](#422-model-fine-tuning-using-validated-durgs)
      - [4.2.3 Binding analysis of potential drugs using AutoDock](#423-binding-analysis-of-potential-drugs-using-autodock)
      - [4.2.4 Example drug candidates and their validated drug templates](#424-example-drug-candidates-and-their-validated-drug-templates)
  - [5. References](#5-references)
  - [Contact](#contact)

Empolyed python, Bert, AWS EC2, docker, lambda, crontab and Event bridge, I build a prediction system that will automatically download reddit comments about bitcoin, sentimental analysis of the comments, then use these sentiment data to predict bitcoin price, and update the result daily to a dashboard here: [dashboard](http://18.224.251.221:8080/)

## 1. Introduction

### 1.1 Zika Virus

Zika virus was first reported in the Zika Forest of Uganda in 1947 among nonhuman primates. Zika virus (ZIKV) and dengue virus (DENV) are closely related flaviviruses that are transmitted by Aedis aegypti, the mosquito vector, and with overlapping geographical distributions. While most ZIKV infections are asymptomatic, they cause a similar immune response and symptoms including fever and body pain. The most well known symptoms of ZIKV infection is in pregnant women, which pose a significant risk to the developing embryo, with microcephaly and other adverse outcomes.

<p align="center">
<img src="/imgs/Structure-of-Zika-Virus.jpeg">
<br>
<em>Zika Virus</em></p>



### 1.2 Proteomics

Proteomics is the large-scale study of proteomes. A proteome is a set of proteins produced in an organism, system, or biological context. Proteomics enables the identification of ever-increasing numbers of proteins. This varies with time and distinct requirements, or stresses, that a cell or organism undergoes.

## 1.3 Deep Learning in Drug Discovery

Drug discovery and development pipelines are long, complex and depend on numerous factors. Machine learning (ML) approaches provide a set of tools that can improve discovery and decision making for well-specified questions with abundant, high-quality data. Opportunities to apply ML occur in all stages of drug discovery. Examples include target validation, identification of prognostic biomarkers and analysis of digital pathology data in clinical trials. 

<p align="center">
<img src="/imgs/lstm_chem.jpg">
<br>
<em>LSTM-based drug generation</em></p>

## 2. Pipeline

I will use PushshiftAPI from psaw package to scrape comments regarding bitcoin from reddit.


Here shows how the scraped comment data looks like:
<p align="center">
<img src="/imgs/PipeLine.png">
<br>
<em>Pipe Line</em></p>

## 3. Differentially expressed proteins

I built an [Online bitcoin comments sentiment analyzer](http://18.118.15.97:8501/) using [Streamlit](https://streamlit.io/) running the trained model. You can input any comments about Bitcoin, the API will do the sentiment analysis for you.

<p align="center">
<img src="/imgs/Fig1.png">
<br>
<em>Differentially expressed Proteins</em></p>


## 4. Training and generation phase of LSTM

### 4.1. Create an Amazon SageMaker notebook instance

Drugs datasets used in this project are from two database: Moses and ChEMBL. Together these two data sets represent about 2.5 million smiles.

The preprocess steps includes removing duplicates, salts, stereochemical information, nucleic acids and long peptides.

### 4.2. Training and Generation
#### 4.2.1 Load the pre-trained Bert model and tokenizer

```python
def main():
    config = process_config(CONFIG_FILE)

    # create the experiments dirs
    create_dirs(
        [config.exp_dir, config.tensorboard_log_dir, config.checkpoint_dir])

    #Create the data generator.
    train_dl = DataLoader(config, data_type='train')
    valid_dl = copy(train_dl)
    valid_dl.data_type = 'valid'

    #Create the model.
    modeler = LSTMChem(config, session='train')

    #Create the trainer.
    trainer = LSTMChemTrainer(modeler, train_dl, valid_dl)

    #Start training the model.
    trainer.train()
 
if __name__ == '__main__':
    main()
```
#### 4.2.2 Model Fine-Tuning using validated durgs
Search the literatures and got experiment validated anti-ZIKV drugs:
Such as:

Niclosamide	
OC1=C(C=C(Cl)C=C1)C(=O)NC1=C(Cl)C=C(C=C1)[N+]([O-])=O

Sofosbuvir
CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3

Add them into the dataset for fine-tune.
```python

modeler = LSTMChem(config, session='finetune')
finetune_dl = DataLoader(config, data_type='finetune')

finetuner = LSTMChemFinetuner(modeler, finetune_dl)
finetuner.finetune()
```
The wandb will generate a parallel coordinates plot, a parameter importance plot, and a scatter plot when you start a W&B Sweep job. 


#### 4.2.3 Binding analysis of potential drugs using AutoDock

Use python library rdkit and meeko to do batch autodocking by the following code.
The rdkit can convert SMILE string into embed molecule. The autodock vina can do the docking using the parameters that user inputs.

```python
lig = rdkit.Chem.MolFromSmiles(fineTuned_smiles)
protonated_lig = rdkit.Chem.AddHs(lig)
rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)  
meeko_prep = meeko.MoleculePreparation()
meeko_prep.prepare(protonated_lig)
lig_pdbqt = meeko_prep.write_pdbqt_string()
v = vina.Vina(sf_name='vina', verbosity=0)
v.set_receptor('target_protein.pdbqt')
v.set_ligand_from_string(lig_pdbqt)
v.compute_vina_maps(center=[-2.029, -53.903,18.744], box_size=[60, 60, 60])
v.dock(exhaustiveness=200, n_poses=5)
output_pdbqt = v.poses(n_poses=5)
```

#### 4.2.4 Example drug candidates and their validated drug templates


<p align="center">
<img src="/imgs/predicted_drugs.png">
<br>
<em>predicted drugs</em></p>

## 5. References

- **Zika Virus**: [https://www.who.int/news-room/fact-sheets/detail/zika-virus](https://www.who.int/news-room/fact-sheets/detail/zika-virus)
- **Introduction of LSTM**: [Understanding LSTM -- a tutorial into Long Short-Term Memory Recurrent Neural Networks](https://arxiv.org/abs/1909.09586)

- **Introduction of LSTM_chem**: [[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://github.com/topazape/LSTM_Chem)]

- **AutoDock**: [https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/autodock](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/autodock)
- **RDKit**: [https://www.rdkit.org/docs/GettingStartedInPython.html](https://www.rdkit.org/docs/GettingStartedInPython.html)

## Contact

- **Author**: Wei Zhang
- **Email**: [zwmc@hotmail.com](zwmc@hotmail.com)
- **Github**: [https://github.com/vveizhang](https://github.com/vveizhang)
- **Linkedin**: [https://www.linkedin.com/in/wei-z-76253523/](https://www.linkedin.com/in/wei-z-76253523/)
