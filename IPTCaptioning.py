# Python image captioning using BLIP model.
# Requires PIL, transformers, torch, and deep_translator.

# If computed thumbnail will be reused or computed again
useImageThumbs = True

# Number of images to be processed at the same time
batchSize = 8

# Max number of threads allowed to enter code between semaphore.acquire() and semaphore.release()
# This can be set if your GPU does not have enough memory to use all threads
maxThreads = None

import os
import sys
import traceback
import threading
import concurrent.futures
import warnings
import json
import time
import io
from collections import OrderedDict

# Suprimir avisos
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# Variáveis globais
enableProp = 'enableImageCaptioning'
enabled = False
configFile = 'ipt_captioning.txt'
targetLanguageProp = 'ipt.captioning.language'
processor = None
model = None
target_language = None
model_lock = threading.RLock()
init_done = False
init_lock = threading.RLock()
semaphore = None
max_image_size = 800  # Limite de tamanho da imagem para economizar memória

def loadModel():
    global processor, model, target_language
    
    model_obj = caseData.getCaseObject('blip_model')
    processor_obj = caseData.getCaseObject('blip_processor')
    
    if model_obj is not None and processor_obj is not None:
        model = model_obj
        processor = processor_obj
        print("Usando modelo BLIP já carregado")
        return model, processor
    
    try:
        from java.lang import System
        iped_root = System.getProperty('iped.root')
        model_path = os.path.join(iped_root, 'models', 'salesforce')
        
        print(f"Carregando modelo de {model_path}")
        
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        # Verificar disponibilidade de GPU
        try:
            import torch
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                print(f"GPU/CUDA disponível: {torch.cuda.get_device_name(0)}")
            else:
                print("GPU/CUDA não disponível, usando CPU")
        except:
            use_gpu = False
            print("Não foi possível verificar disponibilidade de GPU/CUDA, usando CPU")
        
        # Verificar se a biblioteca Accelerate está disponível
        has_accelerate = False
        try:
            import accelerate
            has_accelerate = True
            print("Biblioteca Accelerate disponível")
        except ImportError:
            print("Biblioteca Accelerate não disponível, usando configuração padrão")
        
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
        
        # Mover para GPU se disponível
        if use_gpu:
            model = model.to('cuda')
            print("Modelo atualizado para uso da GPU")
        
        # Armazenar no caseData para reutilização
        caseData.putCaseObject('blip_model', model)
        caseData.putCaseObject('blip_processor', processor)
        
        # Configurar cache
        from java.util.concurrent import ConcurrentHashMap
        cache = ConcurrentHashMap()
        caseData.putCaseObject('caption_cache', cache)

        # Configurar idioma alvo
        target_language = get_target_language()

        print("Modelo BLIP carregado com sucesso")
        return model, processor
    
    except Exception as e:
        error_msg = f"Erro ao carregar modelo: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise RuntimeError(error_msg)

def createSemaphore():
    if maxThreads is None:
        return
    
    global semaphore
    semaphore = caseData.getCaseObject('caption_semaphore')
    
    if semaphore is None:
        from java.util.concurrent import Semaphore
        semaphore = Semaphore(maxThreads)
        caseData.putCaseObject('caption_semaphore', semaphore)
    
    return semaphore

def get_target_language():
    """Obtém o idioma alvo do arquivo de propriedades."""
    try:
        # Obtém o caminho raiz do IPED
        from java.lang import System
        iped_root = System.getProperty('iped.root')

        # Caminho para o arquivo de propriedades
        prop_file_path = os.path.join(iped_root, 'conf', 'ipt_captioning.txt')

        # Valor padrão caso o arquivo não exista ou a propriedade não esteja definida
        default_language = 'pt'

        # Verifica se o arquivo existe
        if not os.path.exists(prop_file_path):
            print("Arquivo de propriedades não encontrado: " + prop_file_path)
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

                    if key == 'ipt.captioning.language':
                        print("Idioma alvo encontrado no arquivo de propriedades: " + value)
                        return value

        # Se a propriedade não for encontrada, retorna o valor padrão
        print("Propriedade ipt.captioning.language não encontrada. Usando valor padrão: " + default_language)
        return default_language

    except Exception as e:
        print("Erro ao ler o arquivo de propriedades: " + str(e))
        return 'pt'  # Valor padrão em caso de erro

def patch_numpy_directly():
    """Aplica patch diretamente no módulo NumPy"""
    try:
        import numpy
        if hasattr(numpy, '__version__') and numpy.__version__.startswith('2.'):
            print(f"Aplicando patch direto no NumPy {numpy.__version__}")
            
            # Configura variáveis de ambiente
            os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
            
            # Desativa funcionalidades problemáticas
            if hasattr(numpy, 'array_api_compat'):
                numpy.array_api_compat = None
            
            # Abordagem mais segura para adicionar _ARRAY_API
            try:
                if not hasattr(numpy, '_ARRAY_API'):
                    setattr(numpy, '_ARRAY_API', None)
            except:
                print("Não foi possível adicionar _ARRAY_API ao NumPy")
            
            print("Patch direto aplicado com sucesso")
            return True
    except Exception as e:
        print(f"Erro ao aplicar patch direto: {str(e)}")
        return False

def isImage(item):
    return item.getMediaType() is not None and item.getMediaType().toString().startswith('image')

def supported(item):
    return item.getLength() is not None and item.getLength() > 0 and (isImage(item))

def convertJavaByteArray(byteArray):
    return bytes(b % 256 for b in byteArray)

def loadRawImage(input):
    from PIL import Image as PilImage
    img = PilImage.open(io.BytesIO(input))
    img = img.convert('RGB')
    
    # Redimensionar se necessário
    if img.width > max_image_size or img.height > max_image_size:
        ratio = min(max_image_size / img.width, max_image_size / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height))
    
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
    
    def getConfigurables(self):
        from iped.engine.config import EnableTaskProperty
        return [EnableTaskProperty(enableProp)]
    
    def init(self, configuration):
        global enabled
        enabled = configuration.getEnableTaskProperty(enableProp)
        print(f"IPTCaptioning habilitado: {enabled}")
        if not enabled:
            return

        # Aplica patch no NumPy antes de qualquer importação
        #patch_numpy_directly()
        
        # Importa as bibliotecas necessárias
        global PilImage, translator
        from PIL import Image as PilImage
        from deep_translator import GoogleTranslator as translator
        
        # Carrega o modelo e cria o semáforo
        loadModel()
        createSemaphore()
    
    def finish(self):
        num_finishes = caseData.getCaseObject('caption_num_finishes')
        if num_finishes is None:
            num_finishes = 0
        
        num_finishes += 1
        caseData.putCaseObject('caption_num_finishes', num_finishes)
        
        # Exibe estatísticas de processamento
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            print(f"Tempo médio de processamento por imagem: {avg_time:.2f} segundos")
            print(f"Tempo total de processamento: {sum(self.processing_times):.2f} segundos")
            print(f"Número de imagens processadas: {len(self.processing_times)}")
        
        if num_finishes == numThreads:
            print(f"Processamento de legendas concluído em todos os {numThreads} threads")
    
    def sendToNextTask(self, item):
        if not item.isQueueEnd() and not self.queued:
            javaTask.get().sendToNextTaskSuper(item)
            return
        
        if self.isToProcessBatch(item):
            for i in self.itemList:
                javaTask.get().sendToNextTaskSuper(i)
            
            self.itemList.clear()
            self.imageList.clear()
        
        if item.isQueueEnd():
            javaTask.get().sendToNextTaskSuper(item)
    
    def isToProcessBatch(self, item):
        size = len(self.itemList)
        return size >= batchSize or (size > 0 and item.isQueueEnd())

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
                    item.setExtraAttribute('ipt_image_captioning', caption)
                    return
            
            img = None

            from keras.preprocessing import image

            # Carregar imagem
            if isImage(item) and not useImageThumbs and item.getTempFile() is not None:
                img_path = item.getTempFile().getAbsolutePath()
                img = PilImage.open(img_path).convert('RGB')
                
                # Redimensionar se necessário
                if img.width > max_image_size or img.height > max_image_size:
                    ratio = min(max_image_size / img.width, max_image_size / img.height)
                    new_width = int(img.width * ratio)
                    new_height = int(img.height * ratio)
                    img = img.resize((new_width, new_height))
            
            if isImage(item) and useImageThumbs and item.getExtraAttribute('hasThumb'):
                input_bytes = convertJavaByteArray(item.getThumb())
                img = loadRawImage(input_bytes)
            
            if not item.isQueueEnd():
                if img is None:
                    item.setExtraAttribute('caption_error', 1)
                    return

                x = image.img_to_array(img)
                self.imageList.append(x)
                self.itemList.append(item)
                self.queued = True
        
        except Exception as e:
            item.setExtraAttribute('caption_error', 2)
            print(f"Erro ao processar imagem: {str(e)}\n{traceback.format_exc()}")
            raise e
        
        if self.isToProcessBatch(item):
            processImages(self.imageList, self.itemList, self.processing_times)

def processImages(imageList, itemList, processing_times):
    print(f'Processando lote de {len(imageList)} imagens.')
    start_time = time.time()
    
    try:
        captions = makePredictions(imageList)
        cache = caseData.getCaseObject('caption_cache')
        
        for i in range(len(itemList)):
            caption = captions[i]
            itemList[i].setExtraAttribute('ImageCaptioning', caption)
            
            if itemList[i].getHash() is not None:
                cache.put(itemList[i].getHash(), caption)
            
            print(f"Legenda definida: {caption}")
        
        # Registrar tempo de processamento
        end_time = time.time()
        batch_time = end_time - start_time
        avg_time = batch_time / len(imageList)
        print(f"Tempo médio por imagem no lote: {avg_time:.2f} segundos")
        
        for _ in range(len(imageList)):
            processing_times.append(avg_time)
    
    except Exception as e:
        print(f"Erro ao processar lote de imagens: {str(e)}\n{traceback.format_exc()}")

def makePredictions(image_list):
    try:
        if semaphore is not None:
            semaphore.acquire()
        
        results = []
        
        for img in image_list:
            try:
                import torch
                
                # Processar imagem
                inputs = processor(img, return_tensors="pt")
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                out = model.generate(**inputs, max_length=50)
                caption_en = processor.decode(out[0], skip_special_tokens=True)
                
                # Traduzir se necessário
                if target_language != 'en':
                    try:
                        caption = translator(source='auto', target=target_language).translate(caption_en)
                    except Exception as e:
                        print(f"Erro na tradução, usando caption em inglês: {str(e)}")
                        caption = caption_en
                else:
                    caption = caption_en
                
                results.append(caption)
                
                # Liberar memória
                del inputs
                del out
                
            except Exception as e:
                print(f"Erro ao gerar caption: {str(e)}")
                results.append("Erro ao gerar legenda")
        
        # Forçar coleta de lixo
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        return results
    
    finally:
        if semaphore is not None:
            semaphore.release()