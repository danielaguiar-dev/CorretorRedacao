import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
import google.generativeai as genai
import pytesseract
import numpy as np
import cv2
from PIL import Image

load_dotenv()

API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)

pytesseract.pytesseract.tesseract_cmd = "dependencies/tesseract-ocr/tesseract.exe"


# Função para extrair texto de uma imagem usando Tesseract OCR
def extrair_texto_da_imagem(arquivo):
    """
    Extrai texto de um arquivo PDF ou imagem.

    Parâmetro:
    - arquivo (str): Caminho do arquivo (.pdf ou imagem .png/.jpg)

    Retorna:
    - texto_extraido (str): Texto extraído do arquivo.
    """
    texto_extraido = ""

    if arquivo.lower().endswith(".pdf"):
        # Converter cada página do PDF em imagem
        imagens = convert_from_path(
            arquivo,
            dpi=300,
            poppler_path="dependencies/poppler-24.08.0/Library/bin",
        )

        for img in imagens:
            texto_extraido += processar_imagem(img) + "\n"

    else:
        # Carregar a imagem diretamente
        imagem = cv2.imread(arquivo)
        texto_extraido = processar_imagem(imagem)

    return texto_extraido.strip()


def processar_imagem(imagem):
    """
    Processa uma imagem para melhorar a extração de texto pelo OCR.

    Parâmetro:
    - imagem (PIL.Image ou np.array): Imagem a ser processada.

    Retorna:
    - texto_extraido (str): Texto extraído da imagem.
    """
    # Se for um objeto PIL, converter para array do OpenCV
    if isinstance(imagem, Image.Image):
        imagem = np.array(imagem)

    # Converter para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplicar threshold para realçar o texto
    _, imagem_tratada = cv2.threshold(
        imagem_cinza, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Converter de volta para PIL (caso necessário)
    imagem_pil = Image.fromarray(imagem_tratada)

    # Extrair texto usando Tesseract
    return pytesseract.image_to_string(imagem_pil, lang="por")


# Função para corrigir a redação usando a API da Gemini
def corrigir_redacao(texto_redacao):
    prompt = f"""
    Você é um corretor especializado em redações do ENEM. Corrija e formate o seguinte texto extraído de uma redação, garantindo:
    - Ortografia e gramática corretas.
    - Coesão e coerência textual.
    - Estrutura conforme o modelo dissertativo-argumentativo do ENEM.
    - Uma proposta de intervenção detalhada e viável.
    
    Texto da redação:
    {texto_redacao}

    Retorne apenas a pontuação total e por competência.
    """

    modelo = genai.GenerativeModel("gemini-2.0-flash")
    resposta = modelo.generate_content(prompt)

    return (
        resposta.text if resposta and resposta.text else "Erro ao processar a correção."
    )


# Função principal
def processar_redacao(caminho_imagem):
    print("🔍 Extraindo texto da imagem...")
    texto_redacao = extrair_texto_da_imagem(caminho_imagem)

    if not texto_redacao:
        print("❌ Erro: Nenhum texto foi detectado na imagem.")
        return

    print("\n📝 Texto extraído:\n")
    print(texto_redacao)

    print("\n📘 Corrigindo redação com a API da Gemini...")
    texto_corrigido = corrigir_redacao(texto_redacao)

    print("\n✅ Redação corrigida:\n")
    print(texto_corrigido)


# Caminho da imagem da redação (modifique conforme necessário)
caminho_imagem = "assets/redacao.pdf"
processar_redacao(caminho_imagem)
