import easyocr

class OCR():
    def __init__(self, langugaes=['en']):
        self.reader = easyocr.Reader(langugaes)
    def Scan(self, image_path):
        results = self.reader.readtext(image_path)
        print(results)

ocr = OCR()

ocr.Scan('/Users/ali/Desktop/RAG/handwriting-2026-04-27-18-40-12.png')