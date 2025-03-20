import os
import sys
import subprocess
import importlib.util
import traceback
from pathlib import Path

enabled = False

def isImage(item):
    return item.getMediaType() is not None and item.getMediaType().toString().startswith('image')

def supported(item):
    return item.getLength() is not None and item.getLength() > 0 and (isImage(item))

class IPTCaptioning:
    def __init__(self):
        self.enableProp = 'enableImageCaptioning'
        self.iped_root = ''

    def isEnabled(self):
        return enabled

    def processQueueEnd(self):
        return True

    def getConfigurables(self):
        from iped.engine.config import EnableTaskProperty
        return [EnableTaskProperty(self.enableProp)]

    def init(self, configuration):
        global enabled
        enabled = configuration.getEnableTaskProperty(self.enableProp)
        self.iped_root = self.get_ipedroot_path()
        if not self.isEnabled():
            logger.info("Módulo IPTCaptioing esta desabilitado..")
            return
        # Importa as bibliotecas necessárias
        logger.info("Importando as bibliotecas..")
        global pilImage, translator, blipProcessor, blipForConditional
        from PIL import Image as pilImage
        from deep_translator import GoogleTranslator as translator
        from transformers import BlipProcessor as blipProcessor
        from transformers import BlipForConditionalGeneration as blipForConditional
        self.initialize()

    def finish(self):
        num_finishes = caseData.getCaseObject('num_finishes')
        if num_finishes is None:
            num_finishes = 0;
        num_finishes += 1
        caseData.putCaseObject('num_finishes', num_finishes)


    def get_ipedroot_path(self):
        # Obtém o caminho raiz do IPED
        try:
            from java.lang import System
            self.iped_root = System.getProperty('iped.root')
            return self.iped_root
        except:
            logger.info("Módulo IPTCaptioing não conseguiu obter o path do IPED_ROOT..")

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
            logger.info("executando processo de inicialização do IPTCaptioning...")
            global processor, model, target_language
            model = caseData.getCaseObject('ipt_captioning_model')
            processor = caseData.getCaseObject('ipt_captioning_processor')
            target_language = caseData.getCaseObject('ipt_captioning_target_language')
            if processor is None or model is None:
                # Obtém o caminho do modelo
                logger.info("Obtendo caminho do modelo...")
                model_path = self.get_model_path()
                if model_path is None:
                    raise RuntimeError("Não foi possível determinar o caminho do modelo. Verifique se IPED_ROOT está definido corretamente.")

                logger.info("Caminho do modelo localizado em: " + model_path)

                # Carrega o processador e o modelo
                logger.info("Carregando o processador e o modelo...")
                try:
                    processor = blipProcessor.from_pretrained(model_path)
                    caseData.putCaseObject('ipt_captioning_processor', processor)
                    logger.info("Processador carregado com sucesso")
                    model = blipForConditional.from_pretrained(model_path)
                    caseData.putCaseObject('ipt_captioning_model', model)
                    logger.info("Modelo carregado com sucesso")
                except Exception as e:
                    error_msg = "Erro ao carregar processador ou modelo: "+str(e)
                    logger.info(error_msg)
                    raise RuntimeError(error_msg)

                # Atualiza o idioma alvo
                logger.info("Configurando idioma alvo...")

                target_language = self.get_target_language()
                caseData.putCaseObject('ipt_captioning_target_language', target_language)
                logger.info("Idioma alvo configurado: " + target_language)

                logger.info("Inicialização concluída com sucesso!")
            else:
                logger.info("Processador e modelo já estão inicializados")

        except Exception as e:
            error_msg = "Erro durante a inicialização: "+str(e)+"\nStack trace:\n"+traceback.format_exc()
            logger.info(error_msg)
            raise RuntimeError(error_msg)

    def process(self, item):

        if not item.isQueueEnd() and not supported(item):
            return

        if item.getTempFile() is None:
            return

        try:
            # Abre e processa a imagem
            try:
                # Obtém o caminho da imagem do item
                image_path = item.getTempFile().getAbsolutePath()
                logger.info("Iniciando processamento da imagem: "+image_path)
                raw_image = pilImage.open(image_path).convert('RGB')
            except Exception as e:
                raise RuntimeError("Erro ao abrir a imagem: "+str(e))

            # Gera a legenda
            try:
                logger.info("Gerando legenda...")
                inputs = processor(raw_image, return_tensors="pt")
                out = model.generate(**inputs)
                caption_en = processor.decode(out[0], skip_special_tokens=True)
                logger.info("Legenda gerada em inglês: "+caption_en)
            except Exception as e:
                raise RuntimeError("Erro na geração da legenda: "+str(e))

            # Traduz a legenda se necessário
            if target_language != 'en':
                try:
                    logger.info("Traduzindo legenda para " + target_language)
                    caption_translated = translator(source='auto', target=target_language).translate(caption_en)
                    logger.info("Legenda traduzida: "+caption_translated)
                    item.setExtraAttribute('ipt_image_captioning', caption_translated)
                except Exception as e:
                    raise RuntimeError("Erro na tradução da legenda: "+str(e))
            else:
                item.setExtraAttribute('ipt_image_captioning', caption_en)

        except Exception as e:
            error_message = "Erro ao processar a imagem: "+str(e)+"\nStack trace:\n"+traceback.format_exc()
            logger.info(error_message)
            return error_message
        finally:
            # Cleanup if needed
            pass

