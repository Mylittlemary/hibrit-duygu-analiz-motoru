from transformers import pipeline

class NLPEngine:
    def __init__(self):
        print("Sistem Yapay Zeka (BERT) Modeli üzerinden başlatılıyor...")
        model_name = "dbmdz/bert-base-turkish-cased"
        
        try:
            self.classifier = pipeline("sentiment-analysis", model=model_name)
            print("BAŞARILI: Yapay zeka modeli yüklendi.")
        except Exception as e:
            print(f"Hata: {e}")
            self.classifier = None

    def analiz_et(self, metin):
        metin_low = metin.lower()
        
        # --- ÖNCE KELİME KONTROLÜ (Garantili Sonuç) ---
        if any(w in metin_low for w in ["mutlu", "iyi", "harika", "seviyorum"]):
            return [{"label": "positive", "score": 0.99}]
        
        if any(w in metin_low for w in ["mutsuz", "korku", "endişe", "kötü", "korkuyorum"]):
            return [{"label": "negative", "score": 0.99}]
            
        # --- EĞER BELİRSİZSE YAPAY ZEKA MODELİNİ KULLAN ---
        if self.classifier:
            return self.classifier(metin)
            
        return [{"label": "neutral", "score": 0.50}]