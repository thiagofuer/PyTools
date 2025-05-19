
# IPTCaptioning

**IPTCaptioning** é um script de análise automatizada de imagens que integra um modelo de *image captioning* (legenda automática) ao sistema de investigação forense digital [IPED](https://github.com/sepinf-inc/IPED), utilizando o modelo [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) da Salesforce.

O script gera automaticamente descrições para imagens e, opcionalmente, traduz as legendas para outro idioma.

---

## 🧠 Funcionalidades

- Geração de legendas para imagens usando o modelo BLIP.
- Suporte a **captioning condicional**, com prompt personalizado.
- Tradução automática das legendas para diversos idiomas via `deep-translator`.
- Integração nativa com o processamento do IPED.
- Compatível com CPU e GPU (via CUDA).

---

## ⚙️ Integração com o IPED

### 🧩 Pré-requisitos

- IPED instalado (versão compatível com tasks em Python).
- Python 3.9+ instalado.
- Ambiente com `pip` e acesso à internet para instalação de dependências.
- Download local do modelo BLIP.

---

## 🛠️ Configuração

Siga os passos abaixo para instalar o `IPTCaptioning.py` no seu IPED:

### 1. Editar o arquivo `IPEDConfig.txt`

Adicione a seguinte linha:

```ini
enableImageCaptioning=true
```

---

### 2. Editar o arquivo `IPED_ROOT/conf/TaskInstaller.xml`

Adicione a seguinte tag dentro da lista de tasks:

```xml
<task script="IPTCaptioning.py"></task>
```

---

### 3. Baixar o modelo BLIP localmente

1. Crie o diretório abaixo (se ainda não existir):

```bash
IPED_ROOT/models/salesforce
```

2. Acesse: [https://huggingface.co/Salesforce/blip-image-captioning-base/tree/main](https://huggingface.co/Salesforce/blip-image-captioning-base/tree/main)

3. Baixe todos os arquivos listados (como `config.json`, `pytorch_model.bin`, `preprocessor_config.json`, etc.)

4. Salve-os dentro do diretório criado:

```
IPED_ROOT/models/salesforce/
```

---

### 4. Instalar as dependências Python

Instale os pacotes abaixo (de preferência em um ambiente virtual):

```bash
pip install numpy>=1.17,<2.0.0
pip install accelerate
pip install pillow
pip install deep-translator
pip install transformers
pip install torch==2.1.0 torchvision torchaudio
```

---

### 5. Salvar o arquivo de configuração `IPTCaptioningConfig.txt`

1. Baixe o arquivo `IPTCaptioningConfig.txt` (disponível neste repositório).
2. Salve-o no diretório:

```
IPED_ROOT/conf/
```

Você pode personalizar as seguintes propriedades:

```ini
targetlanguage=pt            # Idioma de tradução (ex: pt, en, es)
showLog=true                 # Habilita ou desabilita a exibição de logs
captionPrompt=uma foto de   # Prompt condicional (opcional - Deixe em branco se nao quiser usar)
maxLength=70                # Tamanho máximo da legenda
```

---

### 6. Salvar o script `IPTCaptioning.py`

1. Baixe o arquivo `IPTCaptioning.py` (disponível neste repositório).
2. Salve-o no diretório:

```
IPED_ROOT/scripts/tasks/
```

---

## 🚀 Execução

Após as configurações, basta rodar o IPED normalmente. As imagens processadas receberão uma legenda automática como atributo extra (`ImageCaptioning`), que pode ser visualizado no módulo de análise do IPED (IPED-SearchApp.exe).

---

## 📄 Licença do Modelo BLIP

Este projeto utiliza o modelo [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base), licenciado sob a licença **BSD-3-Clause**. Consulte o aviso de licença no diretório `models/salesforce` para mais detalhes.

---

### 🔧 Bibliotecas utilizadas

- [deep-translator](https://github.com/nidhaloff/deep-translator) — biblioteca para tradução automática entre idiomas. Licença MIT.
