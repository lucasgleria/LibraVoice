# Projeto de Extensão - Estácio
## | Objetivos principais

### | Objetivos Técnicos
* **Desenvolvimento de Software**
    * Criar um sistema de reconhecimento de Libras via câmera
    * Implementar conversão de gestos para texto
    * Desenvolver funcionalidade de conversão de texto para áudio
    * Garantir precisão na detecção e tradução dos sinais


### | Objetivos Teóricos
* **Metodológicos**
    * Aplicar os princípios do Scrum no desenvolvimento do projeto
    * Implementar ciclos de SPRINTS quinzenais
    * Utilizar cerimônias ágeis (planning, daily, review, retrospective)
    * Desenvolver competências em gestão ágil de projetos

* **Acadêmicos**
    * Aprofundar conhecimentos em desenvolvimento de software
    * Compreender a aplicação prática de métodos ágeis
    * Estudar técnicas de visão computacional e processamento de linguagem


### | Objetivos Práticos
* **Sociais**
    * Promover inclusão comunicacional entre pessoas surdas e cegas
    * Facilitar o acesso ao mercado de trabalho para PcD
    * Melhorar o acesso à educação e serviços básicos
    * Impactar positivamente entre 20 a 30 pessoas na fase inicial

* **Organizacionais**
    * Estabelecer processo de desenvolvimento organizado e previsível
    * Reduzir retrabalho e atrasos no desenvolvimento
    * Implementar ciclos de feedback com usuários reais
    * Manter documentação clara e estruturada do projeto


### | Objetivo Final
Entregar o protótipo de um software funcional e acessível que sirva como ponte de comunicação entre pessoas com diferentes necessidades, desenvolvido através de metodologias ágeis e validado por usuários reais.


## | O Projeto: LibraVoice.

Neste repositório está o projeto de extensão em python, desenvolvido por mim.

* Público Alvo: _Bancada Avaliativa_
* Tema do software: Tecnologia que traduz Libras em voz
* Disponiblizamos: Acesso completo.

## | O que encontrarei nesse software?
* Arquivo requirements.txt

    * Funcionalidade:
    Lista de todas as bibliotecas utilizadas.
    
### | Pasta: app

* Arquivos nomeados data_collector...py

    * Funcionalidade:
    Coletar dados, foram os scripts utilizados para treinar o programa.

* Arquivo tts.py

    * Funcionalidade:
    Converter o texto para áudio.

* Arquivo main.py

    * Funcionalidade:
    Script raiz do programa, ele captura imagem, converte em texto e devolve uma resposta em áudio utilizando intligencia artificial.
    
### | Pasta: datasets

* Pasta data

    * Funcionalidade:
    Armazenar dados pré-treinados estáticos.

* Pasta data-seq

    * Funcionalidade:
    Armazenar dados pré-treinados dinâmicos.

### | Pasta: models

* Arquivos model.py e train_sequence_model.py

    * Funcionalidade:
    Scripts de treinamento de modelo, foram utilizados para gerar o arquivo de modelo model.pkl, sequence_model.h5 e label_map.npy.
    
    
## | Detalhes

### | Bibliotecas utilizadas:
> absl-py==2.2.2\
> attrs==25.3.0\
> cffi==1.17.1\
> contourpy==1.3.1\
> cycler==0.12.1\
> flatbuffers==25.2.10\
> fonttools==4.57.0\
> jax==0.5.3\
> jaxlib==0.5.3\
> kiwisolver==1.4.8\
> matplotlib==3.10.1\
> mediapipe==0.10.21\
> ml_dtypes==0.5.1\
> numpy==1.26.4\
> opencv-contrib-python==4.11.0.86\
> opencv-python==4.11.0.86\
> opt_einsum==3.4.0\
> packaging==24.2\
> pillow==11.1.0\
> protobuf==4.25.6\
> pycparser==2.22\
> pyparsing==3.2.3\
> python-dateutil==2.9.0.post0\
> scipy==1.15.2\
> sentencepiece==0.2.0\
> six==1.17.0\
> sounddevice==0.5.1


## | Tecnologias Utilizadas

**Desenvolvimento da Aplicação**

[![Python](https://img.shields.io/badge/python-FFFFFF?style=for-the-badge&logo=python&logoColor=000)](https://python.com/)
[![tensorflow](https://img.shields.io/badge/tensorflow-FFFFFF?style=for-the-badge&logo=tensorflow&logoColor=000)](https://www.tensorflow.org/?hl=pt-br)
[![mediapipe](https://img.shields.io/badge/mediapipe-FFFFFF?style=for-the-badge&logo=mediapipe&logoColor=000)](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=pt-br)
[![numpy](https://img.shields.io/badge/numpy-FFFFFF?style=for-the-badge&logo=numpy&logoColor=000)](https://numpy.org)
[![Opencv](https://img.shields.io/badge/Opencv(cv2)-FFFFFF?style=for-the-badge&logo=opencv&logoColor=000)](https://docs.opencv.org/4.x/)

**Manipulação, controle e hospedagem de dados e versões**

[![GitHub](https://img.shields.io/badge/GitHub-FFFFFF?style=for-the-badge&logo=github&logoColor=000)](https://docs.github.com/)
[![Git](https://img.shields.io/badge/Git-FFFFFF?style=for-the-badge&logo=git&logoColor=000)](https://git-scm.com/doc)


## | Equipe

* [Lucas Gomes Leria](https://www.linkedin.com/in/lucasleria/) - Desenvolvedor UX e Programador Fullstack.

## | Fontes

* [Estácio](https://estacio.br/) - Conteúdo avançado em java; Proposta inicial do projeto extensionista. 

* [Tensorflow](https://www.tensorflow.org/?hl=pt-br) - Documentação oficial do Tensorflow utilizada para a utilização da principal Inteligencia artificial da aplicação.

* [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=pt-br) - Documentação oficial do Google mediapipe utilizada para o desenvolvimento da aplicação.

* [Numpy](https://numpy.org) - Documentação oficial do Numpy utilizada para o desenvolvimento da aplicação.

* [OpenCv](https://docs.opencv.org/4.x/) - Documentação oficial da segunda IA utilizada para o desenvolvimento da aplicação.

## | Anexos

* [Meu artigo](images/artigo/945b071300c316412536cb8466c390fce565f5c6fdcca48e7cf0286eb43889c3-1745625477476.pdf)