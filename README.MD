# 🤟 LibraVoice  

**Sistema de tradução de Libras para voz desenvolvido em Python**  

[![Licença](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.1.2-green.svg)]()
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow.svg)]()
[![Deploy](https://img.shields.io/badge/deploy-inactive-red.svg)]()

## 📌 Sumário

1. [Sobre o Projeto](#-sobre-o-projeto)  
2. [Objetivos](#-objetivos)  
3. [Tecnologias](#-tecnologias)  
4. [Funcionalidades](#-funcionalidades)  
5. [Pré-requisitos](#%EF%B8%8F-pré-requisitos)  
6. [Instalação](#%EF%B8%8F-instalação)  
7. [Como utilizar](#-como-utilizar)
8. [Estrutura do Projeto](#-estrutura-do-projeto)
9. [Contribuição](#-contribuição)  
10. [Licença](#-licença)  
11. [Contato](#-contato)  
12. [Recursos Adicionais](#-recursos-adicionais)  

## 💻 Sobre o Projeto  

O **LibraVoice** é um projeto de extensão acadêmica que utiliza visão computacional e processamento de linguagem natural para criar uma ponte comunicacional entre pessoas surdas e ouvintes

- **Motivação**: Carência social de sistemas faciliadores de comunicação
- **Público-alvo**: Bancada Avaliativa, Estudantes de Programação, Comunidade de Desenvolvimento e PCDs 
- **Problema resolvido**: Barreira de comunicação entre surdos e ouvintes
- **Diferencial**: Conversão simultânea para texto e voz  
- **Metodologia**: Desenvolvimento ágil com Scrum  

## 🎯 Objetivos  

### 🛠️ Técnicos  
- Desenvolver sistema de reconhecimento de gestos em tempo real  
- Implementar pipeline completo: gesto → texto → voz  
- Alcançar precisão mínima de 85% na detecção  

### 📚 Acadêmicos  
- Aplicar conceitos de visão computacional e IA
- Implementar metodologias ágeis na prática  
- Documentar processo de desenvolvimento  

### 🌍 Sociais  
- Promover inclusão de pessoas com deficiência auditiva  
- Facilitar acesso a serviços básicos e educação  
- Impactar inicialmente 20-30 usuários  

## 🚀 Tecnologias  

**Núcleo do Sistema**  
- Python
- Matplotlib
- Tensorflow
- MediaPipe
- Joblib

**Processamento de Imagem**  
- OpenCV
- Pillow
- Numpy

## ✨ Funcionalidades  

- ✅ **Reconhecimento de Libras:** Captura de gestos via câmera e conversão em tempo real para texto em tempo real. 
- ✅ **Conversão Texto-Áudio (TTS):** Transformação do texto reconhecido em fala, permitindo a comunicação auditiva.
- ✅ **Scripts de Coleta de Dados:** Ferramentas para a coleta e preparação de dados, essenciais para o treinamento e aprimoramento dos modelos de IA.
- ✅ **Modelos de IA Pré-treinados:** Utilização de dados estáticos e dinâmicos para garantir o funcionamento preciso da inteligência artificial.
- ✅ **Scripts de Treinamento de Modelo:** Ferramentas para a geração e atualização dos modelos de IA (model.pkl, sequence_model.h5, label_map.npy).

## ⚙️ Pré-requisitos  

Para executar o projeto localmente, você precisará de:

- Python: Uma versão compatível (recomenda-se a versão 3.9.9 ou mais recente).
- Câmera: Uma câmera funcional conectada ao seu computador para a captura de gestos.
- Bibliotecas Python: Todas as dependências listadas no requirements.txt.
- Uma IDE (Visual Studio Code, PyCharm, etc.) para desenvolvimento e execução.
- Armazenamento livre mínimo: 4GB+ RAM e 2GB+ disco   

## 🛠️ Instalação  

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/LibraVoice.git
```

2. Navegue até o diretório do projeto:
```bash
cd LibraVoice
```

3. (Opcional recomendável) Crie e ative um ambiente virtual:
> É altamente recomendado usar um ambiente virtual para gerenciar as dependências.
```bash
# No Windows
python -m venv venv
.\venv\Scripts\activate
# ou # No macOS/Linux
source venv/bin/activate
```

4. Instale as dependências:
```bash
pip install -r requirements.txt
```

## ❗ Como Utilizar

1. Inicie o programa:
```bash
python app/main.py
```

2. Interaja com a câmera:
Posicione-se em frente à câmera para que o sistema possa capturar seus gestos em Libras.

3. Observe a tradução:
O sistema exibirá a tradução em texto e reproduzirá a resposta em áudio, utilizando inteligência artificial.
> Ps: Apenas algumas Letras do alfabeto foram treinadas nesse modelo. (L,U,C,A,S,J,W)
### ▶️ Demonstração

![Exemplo](https://raw.githubusercontent.com/gist/TheJLifeX/74958cc59db477a91837244ff598ef4a/raw/088f3995801c58f79f0a79086f1cd4cc176396d3/00-hand-gesture-recognition.gif)

*(GIF meramente ilustrativo)*  

## 📂 Estrutura do Projeto  

```plaintext
LibraVoice/
├── app/                    # Contém a lógica principal do software
│   ├── data_collector...py # Scripts para coleta de dados de treinamento
│   ├── tts.py              # Script para conversão de texto para áudio (Text-to-Speech)
│   └── main.py             # Script principal: captura imagem, converte para texto e gera áudio
├── datasets/               # Armazena dados utilizados pelos modelos
│   ├── data/               # Dados pré-treinados estáticos
│   └── data-seq/           # Dados pré-treinados dinâmicos
├── models/                 # Contém os scripts e arquivos dos modelos de IA
│   ├── model.py            # Script de treinamento do modelo principal
│   ├── train_sequence_model.py # Script de treinamento do modelo de sequência
│   └── (arquivos de modelo gerados: model.pkl, sequence_model.h5, label_map.npy)
├── .gitignore              # Arquivo de ignorar para controle de versão
├── LICENSE                 # Licença MIT
├── README.md               # Este arquivo
└── requirements.txt        # Lista de bibliotecas Python utilizadas
```

## 🤝 Contribuição
Contribuições são bem-vindas! Siga estas etapas:

1. Reporte bugs: Abra uma [issue](https://github.com/lucasgleria/LibraVoice/issues) no GitHub.
2. Sugira melhorias: Envie ideias ou pull requests com novas funcionalidades.
3. Desenvolva:
- Faça um fork do projeto.
- Crie uma branch (git checkout -b feature/nova-funcionalidade).
- Envie um Pull Request.

## 📜 Licença  

Distribuído sob licença MIT. Veja [LICENSE](LICENSE) para mais informações. 

## 📞 Contato
- **Autor**: [Lucas Leria](https://github.com/lucasgleria)
- **LinkedIn**: [lucasgleria](https://www.linkedin.com/in/lucasgleria/)

## 🔎 Recursos Adicionais  

- [Documentação TensorFlow](https://www.tensorflow.org/)  
- [Soluções MediaPipe](https://ai.google.dev/mediapipe)  
- [OpenCV para Python](https://docs.opencv.org/)  
- [Artigo do Projeto](imagens/artigo/artigo.pdf)  

> Projeto acadêmico desenvolvido como trabalho de extensão da [Estácio](https://estacio.br/)
