# Python image captioning using BLIP model.
# Requires PIL, transformers, torch, and deep_translator.
# This file uses the BLIP image captioning model from Salesforce
# https://huggingface.co/Salesforce/blip-image-captioning-base
# Licensed under the BSD 3-Clause License
# Copyright (c) 2022, Salesforce.com, Inc.

import os
import traceback
import io

# Variáveis globais
useImageThumbs = False
enableProp = 'enableImageCaptioning'
enabled = False
configFile = 'IPTCaptioningConfig.txt'
targetLanguageProp = 'targetlanguage'
verbosemodeProp = 'verbosemode'
captionPromptProp = 'captionPrompt'
captioning_resultad_metadada = 'ImageCaptioning'

processor = None
model = None
target_language = None
max_image_size = 800  # Limite de tamanho da imagem para economizar memória

def loadModel():
    global processor, model, target_language, use_gpu

    model_obj = caseData.getCaseObject('blip_model')
    processor_obj = caseData.getCaseObject('blip_processor')

    if model_obj is not None and processor_obj is not None:
        model = model_obj
        processor = processor_obj
        logger.info("[IPTCaptioning] Usando modelo BLIP já carregado")
        return model, processor

    try:
        from java.lang import System
        iped_root = System.getProperty('iped.root')
        model_path = os.path.join(iped_root, 'models', 'salesforce')

        logger.info(f"[IPTCaptioning] Carregando modelo de {model_path}")

        from transformers import BlipProcessor, BlipForConditionalGeneration

        # Verificar disponibilidade de GPU
        try:
            import torch
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                logger.info(f"[IPTCaptioning] GPU/CUDA disponível: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("[IPTCaptioning] GPU/CUDA não disponível, usando CPU")
        except:
            use_gpu = False
            logger.warn("[IPTCaptioning] Não foi possível verificar disponibilidade de GPU/CUDA, usando CPU")

        # Verificar se a biblioteca Accelerate está disponível
        has_accelerate = False
        try:
            import accelerate
            has_accelerate = True
            logger.info("[IPTCaptioning] Biblioteca Accelerate disponível")
        except ImportError:
            logger.info("[IPTCaptioning] Biblioteca Accelerate não disponível, usando configuração padrão")

        # Carregar com configurações otimizadas
        processor = BlipProcessor.from_pretrained(
            model_path,
            use_fast=True  # Usar tokenizador rápido
        )

        # Carregar modelo com ou sem low_cpu_mem_usage dependendo da disponibilidade do Accelerate
        if has_accelerate:
            model = BlipForConditionalGeneration.from_pretrained(
                model_path,
                low_cpu_mem_usage=True  # Reduzir uso de memória
            )
        else:
            model = BlipForConditionalGeneration.from_pretrained(model_path)

        # Usa o CUDA se GPU disponível
        if use_gpu:
            model = model.to('cuda')
            logger.info("[IPTCaptioning] Modelo atualizado para uso da GPU")

        # Armazenar no caseData para reutilização
        caseData.putCaseObject('blip_model', model)
        caseData.putCaseObject('blip_processor', processor)

        # Configurar cache
        from java.util.concurrent import ConcurrentHashMap
        cache = ConcurrentHashMap()
        caseData.putCaseObject('caption_cache', cache)

        # Configurar idioma alvo
        target_language = get_target_language()

        logger.info("[IPTCaptioning] Modelo BLIP carregado com sucesso")
        return model, processor

    except Exception as e:
        error_msg = f"[IPTCaptioning] Erro ao carregar modelo: {str(e)}\n{traceback.format_exc()}"
        logger.warn(error_msg)
        raise RuntimeError(error_msg)

def get_target_language():
    """Obtém o idioma alvo do arquivo de propriedades."""
    try:
        # Obtém o caminho raiz do IPED
        from java.lang import System
        iped_root = System.getProperty('iped.root')

        # Caminho para o arquivo de propriedades
        prop_file_path = os.path.join(iped_root, 'conf', configFile)

        # Valor padrão caso o arquivo não exista ou a propriedade não esteja definida
        default_language = 'pt'

        # Verifica se o arquivo existe
        if not os.path.exists(prop_file_path):
            logger.info("[IPTCaptioning] Arquivo de propriedades não encontrado: " + prop_file_path)
            return default_language

        # Lê o arquivo de propriedades
        with open(prop_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Ignora linhas vazias e comentários
                if not line or line.startswith('#'):
                    continue

                # Procura pela propriedade ipt.captioning.language
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == targetLanguageProp:
                        logger.info("[IPTCaptioning] Idioma alvo encontrado no arquivo de propriedades: " + value)
                        return value

        # Se a propriedade não for encontrada, retorna o valor padrão
        logger.info("[IPTCaptioning] Propriedade "+ targetLanguageProp +" não encontrada. Usando valor padrão: " + default_language)
        return default_language

    except Exception as e:
        logger.warn("[IPTCaptioning] Erro ao ler o arquivo de propriedades: " + str(e))
        return 'pt'  # Valor padrão em caso de erro

def isImage(item):
    return item.getMediaType() is not None and item.getMediaType().toString().startswith('image')

def supported(item):
    return item.getLength() is not None and item.getLength() > 0 and (isImage(item))

def convertJavaByteArray(byteArray):
    return bytes(b % 256 for b in byteArray)

def resize_image_if_needed(img):
    """
    Redimensiona a imagem se ela exceder o tamanho máximo definido.
    Args:
        img: Objeto PIL.Image a ser redimensionado
    Returns:
        PIL.Image: Imagem redimensionada ou original
    """
    if img.width > max_image_size or img.height > max_image_size:
        ratio = min(max_image_size / img.width, max_image_size / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height))
    return img

def loadRawImage(input):
    from PIL import Image as PilImage
    img = PilImage.open(io.BytesIO(input))
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
        self.processing_times = []

    def isEnabled(self):
        return enabled

    def processQueueEnd(self):
        return True

    #Carrega as propriedades do arquivo IPEDConfig.txt
    def getConfigurables(self):
        from iped.engine.config import EnableTaskProperty
        return [EnableTaskProperty(enableProp)]

    def init(self, configuration):
        global enabled
        enabled = configuration.getEnableTaskProperty(enableProp)
        logger.info(f"[IPTCaptioning] habilitado: {enabled}")
        if not enabled:
            return

        # Importa as bibliotecas necessárias
        global PilImage, translator
        from PIL import Image as PilImage
        from deep_translator import GoogleTranslator as translator

        # Carrega o modelo
        loadModel()

    def finish(self):
        self.itemList.clear()
        self.imageList.clear()
        self.processing_times.clear()
        logger.info("[IPTCaptioning] Finish")

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
                    item.setExtraAttribute(captioning_resultad_metadada, caption)
                    return

            img = None

            # Carregar imagem
            if isImage(item) and not useImageThumbs and item.getTempFile() is not None:
                img_path = item.getTempFile().getAbsolutePath()
                img = PilImage.open(img_path).convert('RGB')

            if isImage(item) and useImageThumbs and item.getExtraAttribute('hasThumb'):
                input_bytes = convertJavaByteArray(item.getThumb())
                img = loadRawImage(input_bytes)

            img = resize_image_if_needed(img)
            processImage(img, item)
        except Exception as e:
            logger.warn(f"[IPTCaptioning] Erro ao processar imagem: {str(e)}\n{traceback.format_exc()}")
            raise e

def processImage(image, item):
    try:
        caption = makePredictions(image)
        cache = caseData.getCaseObject('caption_cache')
        item.setExtraAttribute(captioning_resultad_metadada, caption)
        #se chegou ate aqui é porque não tinha o cache do item. Entao adicionamos a legenda ao cache
        if item.getHash() is not None:
            cache.put(item.getHash(), caption)
        logger.info(f"[IPTCaptioning] Legenda definida: {caption}")
    except Exception as e:
        logger.warn(f"[IPTCaptioning] Erro ao processar lote de imagens: {str(e)}\n{traceback.format_exc()}")

def makePredictions(img):
    try:
        inputs = processor(img, return_tensors="pt")
        if(use_gpu):
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        out = model.generate(**inputs, max_length=50)
        caption_en = processor.decode(out[0], skip_special_tokens=True)

        # Traduzir se necessário
        if target_language != 'en':
            try:
                caption = translator(source='auto', target=target_language).translate(caption_en)
            except Exception as e:
                logger.warn(f"[IPTCaptioning] Erro na tradução, usando caption em inglês: {str(e)}")
                caption = caption_en
        else:
            caption = caption_en

        return caption
    except Exception as e:
        logger.warn(f"[IPTCaptioning] Erro ao gerar caption: {str(e)}")
        return "Erro ao gerar legenda"