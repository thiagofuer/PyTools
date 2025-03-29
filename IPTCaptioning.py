import os
import sys
import traceback
import threading
import concurrent.futures
import warnings
import json
import time
from collections import OrderedDict

# Suprimir avisos
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# Variáveis globais
enabled = False
processor = None
model = None
target_language = None
model_lock = threading.RLock()
init_done = False
init_lock = threading.RLock()

# Cache com limite de tamanho e política LRU
class LimitedSizeCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Mover para o final (LRU)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remover o item mais antigo
                self.cache.popitem(last=False)
            self.cache[key] = value

# Inicializar cache
caption_cache = LimitedSizeCache(max_size=1000)

def get_cached_caption(image_path, lang):
    """Cache para legendas já processadas"""
    cache_key = f"{image_path}_{lang}"
    return caption_cache.get(cache_key)

def set_cached_caption(image_path, lang, caption):
    """Armazena legenda no cache"""
    cache_key = f"{image_path}_{lang}"
    caption_cache.set(cache_key, caption)

def isImage(item):
    return item.getMediaType() is not None and item.getMediaType().toString().startswith('image')

def supported(item):
    return item.getLength() is not None and item.getLength() > 0 and (isImage(item))

class IPTCaptioning:
    def __init__(self):
        self.enableProp = 'enableImageCaptioning'
        self.iped_root = ''
        self.batch_size = 8
        self.executor = None
        self.numpy_patched = False
        self.processing_times = []
        self.num_finishes = 0
        self.max_image_size = 800  # Limite de tamanho da imagem para economizar memória

    def isEnabled(self):
        return enabled

    def processQueueEnd(self):
        return True

    def getConfigurables(self):
        from iped.engine.config import EnableTaskProperty
        return [EnableTaskProperty(self.enableProp)]

    def monitor_memory_usage(self):
        """Monitora o uso de memória atual"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            logger.info(f"Uso de memória atual: {memory_usage_mb:.2f} MB")
            return memory_usage_mb
        except:
            return None

    def adjust_batch_size(self):
        """Ajusta o batch_size dinamicamente com base no uso de memória"""
        memory_usage_mb = self.monitor_memory_usage()
        if memory_usage_mb:
            if memory_usage_mb > 4000:  # 4GB
                self.batch_size = max(1, self.batch_size - 1)
                logger.info(f"Reduzindo batch_size para {self.batch_size} devido ao alto uso de memória")
            elif memory_usage_mb < 2000 and self.batch_size < 12:  # 2GB
                self.batch_size = self.batch_size + 1
                logger.info(f"Aumentando batch_size para {self.batch_size} devido ao baixo uso de memória")

    def patch_numpy_directly(self):
        """Aplica patch diretamente no módulo NumPy"""
        try:
            import numpy
            if hasattr(numpy, '__version__') and numpy.__version__.startswith('2.'):
                logger.info(f"Aplicando patch direto no NumPy {numpy.__version__}")

                # Adiciona _ARRAY_API diretamente ao módulo
                if not hasattr(numpy, '_ARRAY_API'):
                    numpy._ARRAY_API = None

                # Configura variáveis de ambiente
                os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

                # Desativa funcionalidades problemáticas
                if hasattr(numpy, 'array_api_compat'):
                    numpy.array_api_compat = None

                self.numpy_patched = True
                logger.info("Patch direto aplicado com sucesso")
                return True
        except Exception as e:
            logger.error(f"Erro ao aplicar patch direto: {str(e)}")
        return False

    def init(self, configuration):
        global enabled, init_done

        enabled = configuration.getEnableTaskProperty(self.enableProp)
        self.iped_root = self.get_ipedroot_path()

        if not self.isEnabled():
            logger.info("Módulo IPTCaptioning está desabilitado..")
            return

        # Aplica patch no NumPy antes de qualquer importação
        self.patch_numpy_directly()

        # Cria o executor para processamento paralelo
        if not self.executor:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size)

        # Importa as bibliotecas necessárias apenas uma vez
        with init_lock:
            if not init_done:
                try:
                    logger.info("Importando as bibliotecas..")
                    global pilImage, translator, blipProcessor, blipForConditional
                    from PIL import Image as pilImage
                    from deep_translator import GoogleTranslator as translator
                    from transformers import BlipProcessor as blipProcessor
                    from transformers import BlipForConditionalGeneration as blipForConditional
                    logger.info("Bibliotecas importadas..")
                    # Inicializa o modelo apenas uma vez
                    self.initialize()
                    init_done = True

                except Exception as e:
                    logger.error(f"Erro ao importar bibliotecas: {str(e)}\nTraceback: {traceback.format_exc()}")
                    return

    def finish(self):
        if self.executor:
            self.executor.shutdown(wait=True)

        # Exibe estatísticas de processamento
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            logger.info(f"Tempo médio de processamento por imagem: {avg_time:.2f} segundos")
            logger.info(f"Tempo total de processamento: {sum(self.processing_times):.2f} segundos")
            logger.info(f"Número de imagens processadas: {len(self.processing_times)}")

        # Incrementa contador local
        self.num_finishes += 1
        logger.info(f"Finalizações: {self.num_finishes}")

    def get_ipedroot_path(self):
        # Obtém o caminho raiz do IPED
        try:
            from java.lang import System
            self.iped_root = System.getProperty('iped.root')
            return self.iped_root
        except:
            logger.info("Módulo IPTCaptioning não conseguiu obter o path do IPED_ROOT..")
            return None

    def get_model_path(self):
        """Obtém o caminho do modelo a partir da propriedade do sistema Java."""
        if self.iped_root is None:
            logger.info("IPED_ROOT não definido. Não é possível determinar o caminho do modelo.")
            return None
        # Define o caminho do modelo
        model_path = os.path.join(self.iped_root, 'models', 'salesforce')
        return model_path

    def get_target_language(self):
        """Obtém o idioma alvo do arquivo de propriedades."""
        try:
            # Obtém o caminho raiz do IPED
            from java.lang import System
            self.iped_root = System.getProperty('iped.root')

            # Caminho para o arquivo de propriedades
            prop_file_path = os.path.join(self.iped_root, 'conf', 'ipt_captioning.txt')

            # Valor padrão caso o arquivo não exista ou a propriedade não esteja definida
            default_language = 'pt'

            # Verifica se o arquivo existe
            if not os.path.exists(prop_file_path):
                logger.info("Arquivo de propriedades não encontrado: " + prop_file_path)
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
                            logger.info("Idioma alvo encontrado no arquivo de propriedades: " + value)
                            return value

            # Se a propriedade não for encontrada, retorna o valor padrão
            logger.info("Propriedade ipt.captioning.language não encontrada. Usando valor padrão: " + default_language)
            return default_language

        except Exception as e:
            logger.info("Erro ao ler o arquivo de propriedades: " + str(e))
            return 'pt'  # Valor padrão em caso de erro

    def initialize(self):
        """Inicializa o processador e o modelo."""
        try:
            logger.info("Executando inicialização do IPTCaptioning...")
            global processor, model, target_language

            model_path = self.get_model_path()
            if model_path is None:
                raise RuntimeError("Caminho do modelo não encontrado")

            logger.info(f"Carregando modelo de {model_path}")

            try:
                with model_lock:
                    # Aplica patch novamente para garantir
                    self.patch_numpy_directly()

                    # Verificar disponibilidade de GPU
                    try:
                        import torch
                        use_gpu = torch.cuda.is_available()
                        if use_gpu:
                            logger.info(f"GPU/CUDA disponível: {torch.cuda.get_device_name(0)}")
                        else:
                            logger.info("GPU/CUDA não disponível, usando CPU")
                    except:
                        use_gpu = False
                        logger.info("Não foi possível verificar disponibilidade de GPU/CUDA, usando CPU")

                    # Verificar se a biblioteca Accelerate está disponível
                    has_accelerate = False
                    try:
                        import accelerate
                        has_accelerate = True
                        logger.info("Biblioteca Accelerate disponível")
                    except ImportError:
                        logger.info("Biblioteca Accelerate não disponível, usando configuração padrão")

                    # Carregar com configurações otimizadas
                    processor = blipProcessor.from_pretrained(
                        model_path,
                        use_fast=True  # Usar tokenizador rápido
                    )

                    # Carregar modelo com ou sem low_cpu_mem_usage dependendo da disponibilidade do Accelerate
                    if has_accelerate:
                        model = blipForConditional.from_pretrained(
                            model_path,
                            low_cpu_mem_usage=True  # Reduzir uso de memória
                        )
                    else:
                        model = blipForConditional.from_pretrained(model_path)

                    # Mover para GPU se disponível
                    if use_gpu:
                        model = model.to('cuda')
                        logger.info("Modelo atualizado para uso da GPU")

                    target_language = self.get_target_language()
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")

            logger.info("Inicialização concluída")

        except Exception as e:
            error_msg = f"Erro na inicialização: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def load_image_with_timeout(self, image_path, timeout=10):
        """Carrega uma imagem com timeout"""
        future = self.executor.submit(pilImage.open, image_path)
        try:
            raw_image = future.result(timeout=timeout)
            return raw_image.convert('RGB')
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout ao carregar imagem: {image_path}")

    def generate_caption_with_timeout(self, raw_image, timeout=30):
        """Gera caption com timeout"""
        def _generate():
            try:
                import torch
                inputs = processor(raw_image, return_tensors="pt")
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                out = model.generate(**inputs, max_length=50)
                caption = processor.decode(out[0], skip_special_tokens=True)
                del inputs
                del out
                return caption
            except Exception as e:
                logger.error(f"Erro ao gerar caption: {str(e)}")
                raise

        future = self.executor.submit(_generate)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Timeout ao gerar caption")

    def translate_with_timeout(self, text, timeout=10):
        """Traduz texto com timeout"""
        future = self.executor.submit(translator(source='auto', target=target_language).translate, text)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Timeout na tradução")

    def process_batch(self, image_paths):
        """Processa um lote de imagens de uma vez"""
        results = {}
        start_time = time.time()

        try:
            # Verificar cache para todas as imagens
            to_process = []
            for path in image_paths:
                cached = get_cached_caption(path, target_language)
                if cached:
                    results[path] = cached
                else:
                    to_process.append(path)

            if not to_process:
                return results

            # Processar imagens não cacheadas
            with model_lock:
                try:
                    import torch

                    # Carregar e preparar todas as imagens
                    raw_images = []
                    for path in to_process:
                        try:
                            img = pilImage.open(path).convert('RGB')

                            # Redimensionar se necessário
                            if img.width > self.max_image_size or img.height > self.max_image_size:
                                ratio = min(self.max_image_size / img.width, self.max_image_size / img.height)
                                new_width = int(img.width * ratio)
                                new_height = int(img.height * ratio)
                                img = img.resize((new_width, new_height))

                            raw_images.append((path, img))
                        except Exception as e:
                            logger.error(f"Erro ao carregar imagem {path}: {str(e)}")

                    # Processar cada imagem
                    for path, img in raw_images:
                        try:
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
                                    logger.warning(f"Erro na tradução, usando caption em inglês: {str(e)}")
                                    caption = caption_en
                            else:
                                caption = caption_en

                            # Armazenar resultado
                            results[path] = caption
                            set_cached_caption(path, target_language, caption)

                            # Liberar memória
                            del inputs
                            del out
                            img.close()
                        except Exception as e:
                            logger.error(f"Erro ao processar imagem {path}: {str(e)}")

                    # Forçar coleta de lixo
                    import gc
                    gc.collect()
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Erro no processamento em lote: {str(e)}")

        except Exception as e:
            logger.error(f"Erro geral no processamento em lote: {str(e)}\n{traceback.format_exc()}")

        # Registrar tempo de processamento
        end_time = time.time()
        batch_time = end_time - start_time
        if to_process:
            avg_time = batch_time / len(to_process)
            logger.info(f"Tempo médio por imagem no lote: {avg_time:.2f} segundos")
            for _ in to_process:
                self.processing_times.append(avg_time)

        return results

    def process_image(self, image_path):
        """Processa uma única imagem com melhor gestão de memória"""
        start_time = time.time()
        try:
            # Verifica cache
            cached_result = get_cached_caption(image_path, target_language)
            if cached_result:
                logger.info(f"Usando caption em cache para {image_path}")
                return cached_result

            # Verificar se o arquivo existe e é acessível
            if not os.path.isfile(image_path) or not os.access(image_path, os.R_OK):
                logger.error(f"Arquivo não existe ou não é acessível: {image_path}")
                return None

            # Verificar tamanho do arquivo
            file_size = os.path.getsize(image_path)
            if file_size > 20 * 1024 * 1024:  # 20MB
                logger.warning(f"Arquivo muito grande ({file_size/1024/1024:.2f} MB), pode causar problemas: {image_path}")

            # Processar com timeout para cada etapa
            try:
                # Carregar imagem com timeout
                load_timeout = 10  # segundos
                raw_image = self.load_image_with_timeout(image_path, timeout=load_timeout)

                # Redimensionar para economizar memória
                if raw_image.width > self.max_image_size or raw_image.height > self.max_image_size:
                    ratio = min(self.max_image_size / raw_image.width, self.max_image_size / raw_image.height)
                    new_width = int(raw_image.width * ratio)
                    new_height = int(raw_image.height * ratio)
                    raw_image = raw_image.resize((new_width, new_height))

                # Processar com timeout
                process_timeout = 30  # segundos
                caption_en = self.generate_caption_with_timeout(raw_image, timeout=process_timeout)

                # Liberar memória
                raw_image.close()

                # Traduzir com timeout
                if target_language != 'en':
                    try:
                        translate_timeout = 10  # segundos
                        caption = self.translate_with_timeout(caption_en, timeout=translate_timeout)
                        logger.info(f"Tradução realizada com sucesso para {target_language}")
                    except Exception as e:
                        logger.warning(f"Erro na tradução, usando caption em inglês: {str(e)}")
                        caption = caption_en
                else:
                    caption = caption_en

                # Forçar coleta de lixo
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

            except TimeoutError as e:
                logger.error(f"Timeout durante o processamento: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Erro durante o processamento: {str(e)}")
                return None

            # Armazena no cache
            set_cached_caption(image_path, target_language, caption)
            logger.info(f"Nova caption gerada e armazenada em cache para {image_path}")

            # Registra o tempo de processamento
            end_time = time.time()
            processing_time = end_time - start_time
            self.processing_times.append(processing_time)

            # Ajustar batch_size com base no uso de memória
            self.adjust_batch_size()

            return caption

        except Exception as e:
            logger.error(f"Erro ao processar {image_path}: {str(e)}\n{traceback.format_exc()}")
            end_time = time.time()
            processing_time = end_time - start_time
            self.processing_times.append(processing_time)
            return None

    def process(self, item):
        if item.isQueueEnd():
            return

        if not supported(item):
            return

        if item.getTempFile() is None:
            return

        try:
            image_path = item.getTempFile().getAbsolutePath()
            logger.info(f"Iniciando processamento da imagem: {image_path}")

            # Verificar se o arquivo existe e é acessível
            if not os.path.isfile(image_path) or not os.access(image_path, os.R_OK):
                error_message = f"Arquivo não existe ou não é acessível: {image_path}"
                logger.error(error_message)
                return error_message

            # Inicializa executor se necessário
            if not self.executor:
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size)

            # Processa imagem
            future = self.executor.submit(self.process_image, image_path)
            caption = future.result(timeout=60)  # Adiciona timeout de 60 segundos

            if caption:
                item.setExtraAttribute('ipt_image_captioning', caption)
                logger.info(f"Legenda definida para {image_path}: {caption}")

        except concurrent.futures.TimeoutError:
            error_message = f"Timeout ao processar a imagem: {image_path}"
            logger.error(error_message)
            return error_message
        except Exception as e:
            error_message = f"Erro ao processar a imagem: {str(e)}\nStack trace:\n{traceback.format_exc()}"
            logger.error(error_message)
            return error_message