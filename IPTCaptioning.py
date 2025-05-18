# Python image captioning using BLIP model.
# Requires PIL, transformers, torch, and deep_translator.
# This file uses the BLIP image captioning model from Salesforce
# https://huggingface.co/Salesforce/blip-image-captioning-base
# Licensed under the BSD 3-Clause License
# Copyright (c) 2022, Salesforce.com, Inc.

import os
import traceback
import io

# Variáveis globais de configuração de propriedades
enableProp = 'enableImageCaptioning'
configFile = 'IPTCaptioningConfig.txt'
targetLanguageProp = 'targetlanguage'
showLogProp = 'showLog'
captionPromptProp = 'captionPrompt'
maxLengthProp = 'maxLength'
captioning_result_metadada = 'ImageCaptioning'

# Variáveis globais de estado e modelo
enabled = False
useImageThumbs = False
processor = None
model = None
use_gpu = False
max_image_size = 800

# Variáveis globais para armazenar os valores das propriedades lidas
target_language = 'pt'
show_log_messages = True
caption_prompt_value = ''
max_caption_length = 70

# --- Wrapper para Logger Condicional ---
def _log_info(message):
    if show_log_messages:
        logger.info(message)

def _log_warn(message):
    if show_log_messages:
        logger.warn(message)

# --- Função para ler os valores do arquivo de propriedades IPTCaptioningConfig.txt  ---
def _get_property_from_config_file(property_name, default_value):
    """
    Lê uma propriedade específica do arquivo de configuração.
    Args:
        property_name (str): O nome da propriedade a ser buscada.
        default_value: O valor a ser retornado se a propriedade não for encontrada ou o arquivo não existir.
    Returns:
        str or type(default_value): O valor da propriedade (como string) ou o valor padrão.
    """
    try:
        from java.lang import System
        iped_root = System.getProperty('iped.root')
        prop_file_path = os.path.join(iped_root, 'conf', configFile)

        if not os.path.exists(prop_file_path):
            # Não usar _log_info aqui, pois show_log_messages pode não estar definido ainda se chamado muito cedo
            print(f"[IPTCaptioning] WARNING: Arquivo de configuração '{configFile}' não encontrado em '{prop_file_path}'. Usando valor padrão '{default_value}' para '{property_name}'.")
            return default_value

        with open(prop_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == property_name:
                        # Não usar _log_info aqui
                        print(f"[IPTCaptioning] INFO: Propriedade '{property_name}' encontrada: '{value}'")
                        return value

        # Não usar _log_info aqui
        print(f"[IPTCaptioning] INFO: Propriedade '{property_name}' não encontrada em '{configFile}'. Usando valor padrão '{default_value}'.")
        return default_value
    except Exception as e:
        # Não usar _log_warn aqui
        print(f"[IPTCaptioning] WARNING: Erro ao ler propriedade '{property_name}' do arquivo '{configFile}': {str(e)}. Usando valor padrão '{default_value}'.")
        return default_value

# --- FUNÇÕES ESPECÍFICAS PARA OBTER CADA PROPRIEDADE ---
def get_target_language_config():
    """Obtém o idioma alvo do arquivo de propriedades."""
    return _get_property_from_config_file(targetLanguageProp, 'pt')

def get_show_log_config():
    """Obtém a configuração de showLog do arquivo de propriedades."""
    value_str = _get_property_from_config_file(showLogProp, 'true') # Padrão é logar
    return value_str.lower() == 'true'

def get_caption_prompt_config():
    """Obtém o prompt de legenda customizado do arquivo de propriedades."""
    return _get_property_from_config_file(captionPromptProp, '') # Padrão é sem prompt

def get_max_caption_length_config(): # Nova função
    default_max_len_str = "70"
    value_str = _get_property_from_config_file(maxLengthProp, default_max_len_str)
    try:
        val_int = int(value_str)
        if val_int <= 0:
            print(f"[IPTCaptioning] WARNING: Valor para '{maxLengthProp}' deve ser positivo: '{value_str}'. Usando padrão {default_max_len_str}.")
            return int(default_max_len_str)
        return val_int
    except ValueError:
        print(f"[IPTCaptioning] WARNING: Valor inválido para '{maxLengthProp}': '{value_str}'. Usando padrão {default_max_len_str}.")
        return int(default_max_len_str)

def loadModel():
    global processor, model, target_language, use_gpu, show_log_messages, caption_prompt_value, max_caption_length # Adicionado max_caption_length

    # Carregar configurações primeiro, pois afetam os logs subsequentes
    target_language = get_target_language_config()
    show_log_messages = get_show_log_config()
    caption_prompt_value = get_caption_prompt_config()
    max_caption_length = get_max_caption_length_config()

    _log_info(f"[IPTCaptioning] Configurações carregadas: target_language='{target_language}', show_log={show_log_messages}, caption_prompt='{caption_prompt_value}', max_caption_length={max_caption_length}") # Adicionado ao log

    model_obj = caseData.getCaseObject('blip_model')
    processor_obj = caseData.getCaseObject('blip_processor')

    if model_obj is not None and processor_obj is not None:
        model = model_obj
        processor = processor_obj
        _log_info("[IPTCaptioning] Usando modelo BLIP já carregado")
        # As configurações já foram lidas acima e estarão disponíveis
        return model, processor

    try:
        from java.lang import System
        iped_root = System.getProperty('iped.root')
        model_path = os.path.join(iped_root, 'models', 'salesforce')

        _log_info(f"[IPTCaptioning] Carregando modelo de {model_path}")

        from transformers import BlipProcessor, BlipForConditionalGeneration

        try:
            import torch
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                _log_info(f"[IPTCaptioning] GPU/CUDA disponível: {torch.cuda.get_device_name(0)}")
            else:
                _log_info("[IPTCaptioning] GPU/CUDA não disponível, usando CPU")
        except Exception as e_torch:
            use_gpu = False
            _log_warn(f"[IPTCaptioning] Não foi possível verificar disponibilidade de GPU/CUDA, usando CPU: {str(e_torch)}")

        has_accelerate = False
        try:
            import accelerate
            has_accelerate = True
            _log_info("[IPTCaptioning] Biblioteca Accelerate disponível")
        except ImportError:
            _log_info("[IPTCaptioning] Biblioteca Accelerate não disponível, usando configuração padrão")

        processor = BlipProcessor.from_pretrained(
            model_path,
            use_fast=True
        )

        if has_accelerate:
            model = BlipForConditionalGeneration.from_pretrained(
                model_path,
                low_cpu_mem_usage=True
            )
        else:
            model = BlipForConditionalGeneration.from_pretrained(model_path)

        if use_gpu:
            model = model.to('cuda')
            _log_info("[IPTCaptioning] Modelo atualizado para uso da GPU")

        caseData.putCaseObject('blip_model', model)
        caseData.putCaseObject('blip_processor', processor)

        from java.util.concurrent import ConcurrentHashMap
        cache = ConcurrentHashMap()
        caseData.putCaseObject('caption_cache', cache)

        # As configurações (target_language, show_log_messages, caption_prompt_value, max_caption_length) já foram definidas no início da função.
        _log_info("[IPTCaptioning] Modelo BLIP carregado com sucesso.")
        return model, processor

    except Exception as e:
        error_msg = f"[IPTCaptioning] Erro ao carregar modelo: {str(e)}\n{traceback.format_exc()}"
        _log_warn(error_msg) # Usa o wrapper de log
        raise RuntimeError(error_msg)

def isImage(item):
    return item.getMediaType() is not None and item.getMediaType().toString().startswith('image')

def supported(item):
    return item.getLength() is not None and item.getLength() > 0 and (isImage(item))

def convertJavaByteArray(byteArray):
    return bytes(b % 256 for b in byteArray)

def resize_image_if_needed(img):
    if img.width > max_image_size or img.height > max_image_size:
        ratio = min(max_image_size / img.width, max_image_size / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height))
        _log_info(f"[IPTCaptioning] Imagem redimensionada para {new_width}x{new_height}")
    return img

def loadRawImage(input_bytes):
    from PIL import Image as PilImage
    img = PilImage.open(io.BytesIO(input_bytes))
    img = img.convert('RGB')
    return img

'''
Main class
'''
class IPTCaptioning:

    def __init__(self):
        self.itemList = []
        self.imageList = []
        self.queued = False

    def isEnabled(self):
        return enabled

    def processQueueEnd(self):
        return True

    def getConfigurables(self):
        from iped.engine.config import EnableTaskProperty
        return [EnableTaskProperty(enableProp)]

    def init(self, configuration):
        global enabled, PilImage, translator

        initial_show_log = get_show_log_config()
        if initial_show_log:
            logger.info(f"[IPTCaptioning] Tentando habilitar com base na configuração: {configuration.getEnableTaskProperty(enableProp)}")

        enabled = configuration.getEnableTaskProperty(enableProp)

        global show_log_messages
        show_log_messages = get_show_log_config()

        _log_info(f"[IPTCaptioning] habilitado: {enabled}")

        if not enabled:
            return

        from PIL import Image as PilImage
        from deep_translator import GoogleTranslator as translator

        loadModel()

    def finish(self):
        self.itemList.clear()
        self.imageList.clear()
        _log_info("[IPTCaptioning] Finish")

    def process(self, item):
        self.queued = False

        if not item.isQueueEnd() and not supported(item):
            return

        try:
            # Verificar se ja existe resultado no cache
            if item.getHash() is not None:
                cache = caseData.getCaseObject('caption_cache')
                caption = cache.get(item.getHash())
                if caption is not None:
                    item.setExtraAttribute(captioning_result_metadada, caption)
                    _log_info(f"[IPTCaptioning] Legenda recuperada do cache para {item.getName()}: {caption}...")
                    return

            img = None

            # Carregar imagem
            if isImage(item) and not useImageThumbs and item.getTempFile() is not None:
                img_path = item.getTempFile().getAbsolutePath()
                img = PilImage.open(img_path).convert('RGB')
                _log_info(f"[IPTCaptioning] Imagem carregada de arquivo temporário: {img_path}")
            elif isImage(item) and useImageThumbs and item.getExtraAttribute('hasThumb'):
                input_bytes = convertJavaByteArray(item.getThumb())
                img = loadRawImage(input_bytes)
                _log_info(f"[IPTCaptioning] Imagem carregada do thumbnail para {item.getName()}")

            if img is None:
                _log_info(f"[IPTCaptioning] Nenhuma imagem válida para processar para o item {item.getName()}")
                return

            img = resize_image_if_needed(img)
            processImage(img, item) # Passa o item para logging e cache
        except Exception as e:
            _log_warn(f"[IPTCaptioning] Erro ao processar imagem {item.getName() if item else 'desconhecida'}: {str(e)}\n{traceback.format_exc()}")

def processImage(image, item):
    try:
        caption = generateCaption(image)
        cache = caseData.getCaseObject('caption_cache')
        item.setExtraAttribute(captioning_result_metadada, caption)
        #se chegou até aqui é porque não tinha o cache do item. Então adicionamos a legenda ao cache
        if item.getHash() is not None:
            cache.put(item.getHash(), caption)

        _log_info(f"[IPTCaptioning] Legenda definida para {item.getName()}: {caption}...")
    except Exception as e:
        _log_warn(f"[IPTCaptioning] Erro ao processar imagem {item.getName() if item else 'desconhecida'} no processImage: {str(e)}\n{traceback.format_exc()}")

def generateCaption(img):
    global caption_prompt_value, max_caption_length
    try:
        # Conditional Image Captioning
        if caption_prompt_value: # Se caption_prompt_value não for nulo ou string vazia
            _log_info(f"[IPTCaptioning] Usando prompt condicional: '{caption_prompt_value}'")
            inputs = processor(images=img, text=caption_prompt_value, return_tensors="pt")
        else:
            inputs = processor(images=img, return_tensors="pt")

        if use_gpu:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        out = model.generate(**inputs, max_length=max_caption_length) # Usa a variável global
        caption_en = processor.decode(out[0], skip_special_tokens=True)
        _log_info(f"[IPTCaptioning] Legenda gerada (EN): {caption_en}...")

        if target_language and target_language.lower() != 'en':
            try:
                translated_caption = translator(source='auto', target=target_language).translate(caption_en)
                _log_info(f"[IPTCaptioning] Legenda traduzida para '{target_language}': {translated_caption}...")
                caption = translated_caption
            except Exception as e_translate:
                _log_warn(f"[IPTCaptioning] Erro na tradução de '{caption_en}...' para '{target_language}', usando legenda em inglês: {str(e_translate)}")
                caption = caption_en
        else:
            caption = caption_en

        return caption
    except Exception as e:
        _log_warn(f"[IPTCaptioning] Erro ao gerar legenda: {str(e)}\n{traceback.format_exc()}")
        return "Erro ao gerar legenda"