# Two-stream-action-recognition-keras
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/127003611.svg)](https://zenodo.org/badge/latestdoi/127003611) 

We use spatial and temporal stream cnn under the Keras framework to reproduce published results on UCF-101 action recognition dataset. This is a project from a research internship at the Machine Intelligence team, IBM Research AI, Almaden Research Center, by Wushi Dong (dongws@uchicago.edu).


## References

*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)

*  [[2] Convolutional Two-Stream Network Fusion for Video Action Recognition](https://github.com/feichtenhofer/twostreamfusion)

*  [[3] Five video classification methods](https://github.com/harvitronix/five-video-classification-methods/blob/master/README.md)

*  [[4] UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402)


## Requisitos

  * numpy
  * tensorflow
  * opencv-python

## Pré-processamento do conjunto de dados

### Spatial input data -> rgb frames (etapa não necessária)

  Realizar o download do conjunto de dados, inicialmente criando o diretório `data` e utilizando os seguintes comandos:
  `cd data && wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate`
  
  Depois extrair com `unrar e UCF101.rar`. O tamanho total do arquivo é de cerca de 6.9G.

### Motion input data -> stacked optical flows (Dados pré-processados)

  É possível realizar o download do conjunto de dados já pré-processado utilizando os seguintes códigos
  ```
  cd data
  mkdir opt_flow
  cd opt_flow
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003
  cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
  unzip ucf101_tvl1_flow.zip
  ```
  Em disco o tamanho total é de 25G.

  Para realizar o pré-processamento novamente é necessário utilizar o código disponível em: https://github.com/feichtenhofer/gpu_flow

## Configuração do treinamento

  ```
  class_limit = None  # int, pode ser 1-101 ou None, indica a quantidade de classes
  opt_flow_len = 10 # número de frames utilizados por vídeo
  image_shape=(224, 224)
  batch_size = 64
  nb_epoch = 2222
  ```

## Treinamento

  `python3 temporal_train.py`


## Testes de validação

  `python3 temporal_valida_train.py`

# Instruções Rede Neural Recorrente

    pip install -q git+https://github.com/tensorflow/docs
    wget -q https://git.io/JGc31 -O ucf101_top5.tar.gz
    tar xf ucf101_top5.tar.gz 
    
Para treinar o modelo, digite:

    python train.py
    
Para realizar a inferência em um vídeo escolhido aleatoriamente, digite:

    python inference.py
    
Um GIF 'animation.gif' do vídeo será salvo para conferir e o resultado sairá no terminal
