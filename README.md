# DeepfakeSII

## Overview

Questo progetto confronta diversi modelli di deep learning per distinguere immagini e video reali da contenuti facciali sintetici o manipolati. Il lavoro comprende l'analisi esplorativa dei dataset, il preprocessing di FaceForensics++, l'addestramento e la valutazione dei modelli, con analisi delle predizioni tramite confusion matrix, confidence analysis, Grad-CAM e t-SNE.

## Report

Il report del progetto è disponibile [qui](Progetto_SII.pdf).

## Codice

`src`: contiene il codice per la gestione dei dataset, modelli, training ed evaluation. [Link al codice](src).

Preparazione del dataset FaceForensics++ per HPC: [Link al codice](create_dataset_script/create_ffpp_data_hpc.py).

Script SBATCH per sottomettere i job su HPC: [Link agli script](scripts_hpc).

EDA e risultati su 140K Real and Fake Faces: [Link al notebook](notebooks/140k.ipynb).

EDA di FaceForensics++: [Link al notebook](notebooks/faceforensics%2B%2B_eda.ipynb).

Risultati su FaceForensics++: [Link al notebook](notebooks/faceforensics%2B%2B_results.ipynb).
