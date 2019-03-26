# Kinetics i3D

- [Kinetics i3D](#kinetics-i3d)
  - [Resumo](#resumo)
  - [Setup](#setup)
    - [Setup inicial](#setup-inicial)
    - [Criação da imagem e container Docker](#cria%C3%A7%C3%A3o-da-imagem-e-container-docker)
  - [Inferência](#infer%C3%AAncia)
    - [Exemplo completo](#exemplo-completo)
  - [Treinamento](#treinamento)
  - [Problemas](#problemas)


## Resumo

Esse repositório contém uma implementação em [Tensorflow](https://www.tensorflow.org/) do modelo I3D, baseado no artigo "[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)" e no código original dos autores, com implementação baseada na framework [Sonnet](https://github.com/deepmind/sonnet) (que é um wrapper de alto nível sobre o Tensorflow), encontrado em: https://github.com/deepmind/kinetics-i3d.


## Setup

### Setup inicial

Primeiramente é necessário instalar os drivers da Nvidia e o nvidia-docker:

- [Instalação dos drivers Nvidia para Ubuntu](http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux)
- [Instalação nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

Para conferir a instalação, executar:

```
nvidia-smi
nvidia-docker --version
```

Se os comandos forem executados sem erros, mostrando informações sobre a placa de vídeo e versões dos drives, os pacotes foram instalados.

### Criação da imagem e container Docker

Dentro da pasta `setup`, crie a imagem docker com as dependências necessárias:

```
docker build -t i3d .
```

Inicie o container Docker executando o script `start.sh`:

```
./start.sh DIRETORIO1 DIRETORIO2 ...
```

Onde `DIRETORIO1 DIRETORIO2 ...` são uma sequência de diretórios (precisa ser o path completo) que serão mapeados para dentro do container. Sem isso não é possível acessar os arquivos do computador dentro do container Docker. Exemplo:

```
./start.sh /var/tmp /home/usuario/Documents
```

Assim, os diretórios `/var/tmp` e `/home/usuario/Documents` estarão disponíveis.


## Inferência

O script `evaluate.py` recebe como parâmetros os pesos pré treinados do modelo I3D e um vídeo, no qual a inferência será realizada, com a opção de mostrar o display do resultado em tempo real e também salvar o vídeo anotado. As cenas do vídeo serão classificadas de acordo com as 400 classes do dataset [Kinetics 400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).


Detalhadamente, os parâmetros são:

- `-i` : Caminho para o vídeo de entrada
- `-rgb` (opcional) : Caminho para o modelo RGB (com extensão `.ckpt`) - o modelo padrão é o `rgb_imagenet`, utilizado no paper
- `-flow` _(opcional)_ : Caminho para o modelo baseado em optical flow (com extensão `.ckpt`)
- `-o` _(opcional)_ : Caminho para o vídeo de saída (com nome e extensão inclusos)
- `-display` _(opcional)_ : Flag que decide se o vídeo deve ser exibido enquanto processado

Exemplo:

```
python evaluate.py -rgb /home/models/rgb_imagenet/model.ckpt -i /home/data/video_entrada.mp4 -o /home/data/video_saida.mp4
```

Os modelos pré treinados podem ser encontrados no repositório original dos criadores do método:

https://github.com/deepmind/kinetics-i3d/tree/master/data/checkpoints

Dentro de cada pasta se encontram alguns arquivos, que representam o modelo gerado, os pesos salvos e os checkpoints de treinamento. Os caminhos que devem ser passados por parâmetro para o script de inferência devem terminar com extensão `.ckpt` (e não `ckpt.index` e `ckpt.meta`, como nos arquivos encontrados nas pastas). As pastas prefixadas com `rgb` representam os modelos em RGB, e as pastas prefixadas por `flow` representam os modelos baseados em optical flow.

### Exemplo completo

Instale os drivers da **Nvidia** e o **nvidia-docker**, assim como descrito acima, obtenha algum vídeo, mova-o para a pasta `/var/tmp` e execute os comandos a seguir:

```
git clone -b i3d http://serv113/gitlab/Oficinas/VideoAnalytics2018

cd VideoAnalytics2018/i3d/setup

docker build -t i3d .

./start.sh /var/tmp

python evaluate.py -i /var/tmp/meu_video.mp4 
```

O script acima irá:

- Baixar este repositório para o diretório atual
- Criar a imagem docker, dentro da pasta `VideoAnalytics2018/i3d/setup`
- Iniciar o container docker mapeando o diretório `/var/tmp`
- Executar o script `evaluate.py`, passando como parâmetro `meu_video.mp4` e o modelo RGB pré treinado do I3D (`rgb_imagenet`)

Se algum display estiver disponível, o vídeo será mostrado com as classificações das cenas no canto superior esquerdo, atualizadas a cada segundo.

Caso desejado, o modelo baseado em optical flow pode também ser executado adicionando a opção: `-flow /var/tmp/kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt` no último comando do script acima. A adição da rede baseada em optical flow resulta em um aumento pequeno na qualidade de classificação (vide paper). Porém, o custo computacional para o cálculo do optical flow é muito alto, o que impede sua execução em tempo real.



## Treinamento

Em progresso...


## Problemas

O intuito do método não é ser usado em vídeos contínuos com diversas cenas. Todos os dados de treinamento e inferência são baseados em vídeos de poucos segundos (no máximo 10), contendo uma das 400 cenas específicas do dataset **Kinetics**. Isso significa que em vídeos longos as informações de cenas passadas serão totalmente descartadas a cada inferência do modelo.