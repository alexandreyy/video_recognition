# R(2+1)D and Mixed-Convolutions for Action Recognition

- [R(2+1)D and Mixed-Convolutions for Action Recognition](#r21d-and-mixed-convolutions-for-action-recognition)
  - [Resumo](#resumo)
  - [Sobre o paper](#sobre-o-paper)
  - [Execução](#execu%C3%A7%C3%A3o)
    - [Requisitos](#requisitos)
    - [Configurando o ambiente](#configurando-o-ambiente)
      - [Build da imagem docker](#build-da-imagem-docker)
      - [Rodando o container](#rodando-o-container)
    - [Treinamento](#treinamento)
      - [Finetuning](#finetuning)
        - [HMDB51](#hmdb51)
      - [Treinamento do zero](#treinamento-do-zero)
    - [Inferência](#infer%C3%AAncia)
  - [Atividades realizadas](#atividades-realizadas)

## Resumo
Aqui encontram-se informações sobre o paper _R(2+1)D and Mixed-Convolutions for Action Recognition_, bem como execução e problemas encontrados durante a reprodução do código do memso.

## Sobre o paper

[TO-DO]

## Execução

### Requisitos
- Docker
- GPU Nvidia
- Caffe2, OpenCV, FFMPEG e outros

### Configurando o ambiente

#### Build da imagem docker
Para executar, deve-se criar um container docker a partir da imagem que encontra-se nesta pasta do repo. Para tal, deve-se _buildar_ a imagem docker utilizando o seguinte comando:

```
nvidia-docker build -t oficina_r21d -f Dockerfile .
```

Vale lembrar que o processo de build é bem longo, uma vez que é necessário dar build em diversos packages diretamente do source.

#### Rodando o container
Ao término do processo, para executar um container deve-se executar os seguintes comandos:

```
xhost +local:root
nvidia-docker run -v PASTA_DATASET:/var/data -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --device /dev/video0 -it oficina_r21d bin/bash
```

Note que há um path em capslock no comando acima. É importante mapear o dataset de uso para dentro do container do docker para que o mesmo possa ser utilizado pelos scripts. Portanto, substitua ```PASTA_DATASET``` pelo path correspondente. **Atenção**: só é possível mapear um volume no docker antes da criação do container. Caso isso não ocorra, será necessário instanciar outro container.

### Treinamento

Ao abrir o container, alterar o ambiente atual do conda para o `r21d`: 

```
source activate r21d
```

#### Finetuning

##### HMDB51 
- Aqui estaremos utilizando o modelo pré-treinado no kinetics, com entrada em RGB. Chamado de ```R2.5D-34```. Mais informações sobre o checkpoint podem ser obtidas [aqui](https://github.com/facebookresearch/VMZ/blob/master/tutorials/models.md). O mesmo já se encontra baixado em ```/home/r2.5d_d34_l32.pkl```, se a imagem foi criada com o dockerfile disponibilizado neste repo.

- O dataset na DIGITS1 encontra-se em ```digits1/DataPublic-Digits1/HMDB51```. Já na DIGITS2 o mesmo se encontra em ```digits2/DataPublic-Digits2-Remote/data/datasets/hmdb51```. Aqui assumimos que o dataset estará mapeado para a pasta ```/var/data/```.

- Copiar os arquivos deste repo ```hmdb51_test_01.csv``` e ```hmdb51_train_01.csv``` para  ```/home/oficina/data/splits/hmdb51```.

- Também substituir os arquivos de script originais que se encontram em ```/home/VMZ/scripts``` para os que estão neste repo. Este passo é opcional, mas diminui o trabalho de quem estiver tentado reproduzir o paper porque já contém os paths resolvidos para datasets/modelos/etc.

- Entrar na pasta raiz do repositório
   ```
   cd /home/VMZ
   ```

- Executar o script para criar o database para treinamento/teste
   ```
   sh scripts/create_hmdb51_lmdb.sh
   ```

- Iniciar o finetuning
   ```
   sh scripts/finetune_hmdb51.sh
   ```

**Nota:** Este passo-a-passo foi seguido a partir do repo oficial do paper. Para mais informações, acesse o tutorial base [aqui](https://github.com/facebookresearch/VMZ/blob/master/tutorials/hmdb51_finetune.md).

#### Treinamento do zero

[TO-DO]

### Inferência

Ao abrir o container, alterar o ambiente atual do conda para o `r21d`: 

```
source activate r21d
```

Rodar o seguinte comando, dentro da pasta `/home/code/src/tools`:

```
GTK_PATH=/usr/lib/gtk-2.0/ python run_inference.py --model_name=r2plus1d --model_depth=34 --clip_length_rgb=32 --gpus=0 --batch_size=1 --load_model_path=../../../r2.5d_d34_l32.pkl?dl=1 --db_type=pickle --features=softmax  --num_labels=400 --input /var/data/video.mp4
```

Os parâmetros que podem e devem ser modificados de acordo com a arquitetura de rede usada são os seguintes:
- `--model_name`: Nome do modelo/arquitetura a ser usado. **Default**: `r2plus1d`
- `--model_depth`: Profundidade, em termos de camadas, do modelo a ser usado. **Default**:34
- `--clip_length_rgb`: Tamanho, em frames, de cada clip a ser amostrado do vídeo. **Default**:32
- `--gpus`: GPUs a serem utilizadas. Note que este script não suporta múltiplas gpus. **Default**:0
- `--load_model_path`: Path para o modelo congelado. 
- `--db_type`: Tipo de binário do modelo
- `--features`: Quais features devem ser buscadas em cada execução da rede. **Default**:softmax
- `--num_labels`: Quantidade de classes exitentes. **Default**: 400 (kinetics)
- `--input`: Vídeo de entrada
- `--labels`: Label para cada classe. **Default**: /home/code/src/labels/kinetics.json (há também, na mesma pasta, o referente ao `hmdb51`)

![alt text](resources/example_inference.png "Exemplo de execução")

## Atividades realizadas
- [x] Entender melhor o paper
- [ ] Explicar e descrever paper
- [x] Reproduzir o ambiente do paper
- [x] Reproduzir treinamento (finetuning) HMDB51
- [x] Reproduzir inferência em dataset de teste do HMDB51
- [ ] Reproduzir treinamento Kinetics
- [ ] Reproduzir treinamento SPORTS
- [ ] Adicionar tensorboard (ou similar) para acompanhar o treinamento
- [x] Criar script para live-demo de inferência
- [ ] Reduzir tamanho da imagem do docker