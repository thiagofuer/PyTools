import os
import sys
import subprocess
import importlib.util
from pathlib import Path

class IPTCaptioning:
    def __init__(self):
        self.processor = None
        self.model = None
        self.target_language = 'pt'  # Valor padrão
        self.enabled = True
        self.enableProp = 'enableIPT'

    def isEnabled(self):
        return self.enabled

    def processQueueEnd(self):
        return True

    def getConfigurables(self):
        from iped.engine.config import EnableTaskProperty
        return [EnableTaskProperty(self.enableProp)]

    def find_python_executable(self):
        """Encontra o executável Python correto com base na estrutura do programa."""
        try:
            # Obtém o caminho raiz do IPED
            from java.lang import System
            iped_root = System.getProperty('iped.root')

            # Caminho para o executável Python baseado na estrutura informada
            if os.name == 'nt':  # Windows
                python_path = os.path.join(iped_root, 'python', 'python.exe')
            else:  # Linux/Mac
                python_path = os.path.join(iped_root, 'python', 'bin', 'python3')

                # Alternativa se o python3 não existir
                if not os.path.exists(python_path):
                    python_path = os.path.join(iped_root, 'python', 'bin', 'python')

            # Verifica se o executável existe
            if os.path.exists(python_path):
                print(f"Executável Python encontrado: {python_path}")
                return python_path

            # Se não encontrar no caminho específico, procura no PATH
            print("Executável Python não encontrado no caminho esperado. Procurando no PATH...")
            python_names = ['python3', 'python']
            for name in python_names:
                try:
                    # Usa 'where' no Windows e 'which' no Linux/Mac
                    if os.name == 'nt':  # Windows
                        result = subprocess.check_output(['where', name], stderr=subprocess.STDOUT)
                    else:  # Linux/Mac
                        result = subprocess.check_output(['which', name], stderr=subprocess.STDOUT)

                    # Retorna o primeiro caminho encontrado
                    paths = result.decode('utf-8').strip().split('\n')
                    if paths:
                        print(f"Executável Python encontrado no PATH: {paths[0]}")
                        return paths[0]
                except subprocess.CalledProcessError:
                    continue

            print("Não foi possível encontrar o executável Python.")
            return None

        except Exception as e:
            print(f"Erro ao procurar o executável Python: {str(e)}")
            return None

    def check_and_install_dependencies(self):
        """Verifica e instala as dependências necessárias."""
        # Encontra o executável Python correto
        python_executable = self.find_python_executable()
        if not python_executable:
            print("ERRO: Não foi possível encontrar o executável Python.")
            return False

        # Primeiro, verifica se o pip está instalado
        pip_installed = False
        try:
            # Tenta importar pip
            import pip
            pip_installed = True
            print("Pip já está instalado.")
        except ImportError:
            print("Pip não está instalado. Tentando instalar...")

            try:
                # Tenta usar ensurepip para instalar o pip
                import ensurepip
                ensurepip.bootstrap(upgrade=True, user=True)
                pip_installed = True
                print("Pip instalado com sucesso usando ensurepip.")
            except ImportError:
                # Se ensurepip não estiver disponível, tenta baixar e executar get-pip.py
                try:
                    import urllib.request
                    import tempfile
                    import os

                    # Cria um diretório temporário
                    with tempfile.TemporaryDirectory() as temp_dir:
                        get_pip_path = os.path.join(temp_dir, "get-pip.py")

                        # Baixa get-pip.py
                        print("Baixando get-pip.py...")
                        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)

                        # Executa get-pip.py
                        print("Executando get-pip.py...")
                        subprocess.check_call([python_executable, get_pip_path, "--user"])

                        pip_installed = True
                        print("Pip instalado com sucesso usando get-pip.py.")
                except Exception as e:
                    print(f"Falha ao instalar pip: {str(e)}")
                    print("Por favor, instale o pip manualmente e tente novamente.")
                    return False

        # Se o pip não pôde ser instalado, retorna False
        if not pip_installed:
            return False

        # Agora que temos pip, instala as dependências necessárias
        required_packages = ['pillow', 'transformers', 'deep_translator', 'torch']

        for package in required_packages:
            try:
                importlib.import_module(package)
                print(f"Pacote {package} já está instalado.")
            except ImportError:
                print(f"Instalando pacote {package}...")
                try:
                    print(f"Instalando pacote {package}...")
                    subprocess.check_call([python_executable, "-m", "pip", "install", package], timeout=300)  # 5 minutos
                    print(f"Pacote {package} instalado com sucesso.")
                except subprocess.TimeoutExpired:
                    print(f"Timeout excedido ao instalar {package}.")
                    return False
                except subprocess.CalledProcessError as e:
                    print(f"Erro ao instalar {package}: {str(e)}")
                    return False

        return True

    def get_model_path(self):
        """Obtém o caminho do modelo a partir da propriedade do sistema Java."""
        # Obtém o caminho raiz do IPED
        try:
            from java.lang import System
            iped_root = System.getProperty('iped.root')
        except:
            # Fallback para um caminho padrão se não conseguir obter
            iped_root = os.path.expanduser("~")

        # Define o caminho do modelo
        model_path = os.path.join(iped_root, 'models', 'salesforce')
        return model_path

    def get_target_language(self):
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
                print(f"Arquivo de propriedades não encontrado: {prop_file_path}")
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
                            print(f"Idioma alvo encontrado no arquivo de propriedades: {value}")
                            return value

            # Se a propriedade não for encontrada, retorna o valor padrão
            print(f"Propriedade ipt.captioning.language não encontrada. Usando valor padrão: {default_language}")
            return default_language

        except Exception as e:
            print(f"Erro ao ler o arquivo de propriedades: {str(e)}")
            return 'pt'  # Valor padrão em caso de erro

    def download_model_if_needed(self, model_path):
        """Baixa o modelo se ele não existir no caminho especificado."""
        # Verifica se o diretório do modelo existe
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            print(f"Diretório do modelo criado: {model_path}")

        # Verifica se o modelo já foi baixado
        model_files = os.listdir(model_path)
        if not model_files or not any(file.endswith('.bin') for file in model_files):
            print("Modelo não encontrado. Baixando...")

            # Importa as bibliotecas necessárias
            from transformers import BlipProcessor, BlipForConditionalGeneration

            # Baixa o modelo e o processador
            print("Baixando o processador e o modelo BLIP...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

            # Salva o modelo e o processador no caminho especificado
            print(f"Salvando o modelo em {model_path}...")
            processor.save_pretrained(model_path)
            model.save_pretrained(model_path)
            print("Modelo baixado e salvo com sucesso!")
        else:
            print(f"Modelo já existe em {model_path}")

    def init(self, configuration):
        self.initialize()

    def initialize(self):
        """Inicializa o processador e o modelo."""
        try:
            print("Iniciando processo de inicialização...")

            if self.processor is None or self.model is None:
                # Verifica e instala dependências
                print("Verificando dependências...")
                dependencies_ok = self.check_and_install_dependencies()
                if not dependencies_ok:
                    error_msg = "Não foi possível instalar as dependências necessárias. Verifique os logs para mais detalhes."
                    print(error_msg)
                    raise RuntimeError(error_msg)

                # Obtém o caminho do modelo
                print("Obtendo caminho do modelo...")
                model_path = self.get_model_path()
                print(f"Caminho do modelo: {model_path}")

                # Baixa o modelo se necessário
                print("Verificando se é necessário baixar o modelo...")
                self.download_model_if_needed(model_path)

                try:
                    # Importa as bibliotecas necessárias
                    print("Importando bibliotecas...")
                    from transformers import BlipProcessor, BlipForConditionalGeneration
                except ImportError as e:
                    error_msg = f"Erro ao importar bibliotecas necessárias: {str(e)}"
                    print(error_msg)
                    raise RuntimeError(error_msg)

                # Carrega o processador e o modelo
                print("Carregando o processador e o modelo...")
                try:
                    self.processor = BlipProcessor.from_pretrained(model_path)
                    print("Processador carregado com sucesso")
                    self.model = BlipForConditionalGeneration.from_pretrained(model_path)
                    print("Modelo carregado com sucesso")
                except Exception as e:
                    error_msg = f"Erro ao carregar processador ou modelo: {str(e)}"
                    print(error_msg)
                    raise RuntimeError(error_msg)

                # Atualiza o idioma alvo
                print("Configurando idioma alvo...")
                self.target_language = self.get_target_language()
                print(f"Idioma alvo configurado: {self.target_language}")

                print("Inicialização concluída com sucesso!")
            else:
                print("Processador e modelo já estão inicializados")

        except Exception as e:
            import traceback
            error_msg = f"Erro durante a inicialização: {str(e)}\nStack trace:\n{traceback.format_exc()}"
            print(error_msg)
            raise RuntimeError(error_msg)

    def process(self, item):
        """Método principal que será chamado pelo IPED para processar uma imagem."""
        try:
            # Verifica se o processador e modelo estão inicializados
            if self.processor is None or self.model is None:
                print("Inicializando processador e modelo...")
                self.initialize()
                if self.processor is None or self.model is None:
                    raise RuntimeError("Falha na inicialização do processador ou modelo")

            # Obtém o caminho da imagem do item
            image_path = item.getPath()
            print(f"Iniciando processamento da imagem: {image_path}")

            # Importa as bibliotecas necessárias
            from PIL import Image
            from deep_translator import GoogleTranslator
            import traceback

            # Abre e processa a imagem
            try:
                raw_image = Image.open(image_path).convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Erro ao abrir a imagem: {str(e)}")

            # Gera a legenda
            try:
                print("Gerando legenda...")
                inputs = self.processor(raw_image, return_tensors="pt")
                out = self.model.generate(**inputs)
                caption_en = self.processor.decode(out[0], skip_special_tokens=True)
                print(f"Legenda gerada em inglês: {caption_en}")
            except Exception as e:
                raise RuntimeError(f"Erro na geração da legenda: {str(e)}")

            # Traduz a legenda se necessário
            if self.target_language != 'en':
                try:
                    print(f"Traduzindo legenda para {self.target_language}...")
                    caption_translated = GoogleTranslator(source='auto', target=self.target_language).translate(caption_en)
                    print(f"Legenda traduzida: {caption_translated}")
                    return caption_translated
                except Exception as e:
                    raise RuntimeError(f"Erro na tradução da legenda: {str(e)}")
            else:
                return caption_en

        except Exception as e:
            error_message = f"Erro ao processar a imagem: {str(e)}\nStack trace:\n{traceback.format_exc()}"
            print(error_message)
            return error_message
        finally:
            # Cleanup if needed
            pass

# Para testes locais
if __name__ == "__main__":
    # Exemplo de uso para testes
    class MockItem:
        def __init__(self, path):
            self.path = path

        def getPath(self):
            return self.path

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "D:\\iped\\exemplo.jpg"

    captioner = IPTCaptioning()
    caption = captioner.process(MockItem(img_path))
    print(f"Legenda: {caption}")