import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

class AudioEngine:
    def sentetik_ses_uret(self, mod="sakin"):
        sr = 22050
        t = np.linspace(0, 2, int(sr * 2))
        if mod == "sakin":
            sinyal = 0.1 * np.sin(2 * np.pi * 440 * t)
        else:
            sinyal = 0.5 * np.sin(2 * np.pi * 880 * t) * np.random.normal(1, 0.5, len(t))
        return sinyal, sr

    def analiz_et(self, sinyal, sr):
        mfccs = librosa.feature.mfcc(y=sinyal, sr=sr, n_mfcc=13)
        enerji = np.sum(sinyal**2) / len(sinyal)
        return {
            "enerji": round(float(enerji), 4),
            "stres_gostergesi": round(float(np.mean(mfccs)), 2)
        }

    # DİKKAT: Bu fonksiyonun 'class' içinde (içeride) olması şart!
    def sesi_gorsellestir(self, dosya_yolu):
        try:
            y, sr = librosa.load(dosya_yolu)
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, color='crimson')
            plt.title(f"Ses Sinyal Analizi: {dosya_yolu}")
            plt.tight_layout()
            print(f"Grafik penceresi açılıyor: {dosya_yolu}")
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

if __name__ == "__main__":
    engine = AudioEngine()
    
    # Önce sayısal test
    for durum in ["sakin", "kaygili"]:
        ses, sr = engine.sentetik_ses_uret(mod=durum)
        sonuc = engine.analiz_et(ses, sr)
        print(f"\nDurum: {durum.upper()} | Enerji: {sonuc['enerji']}")

    # Şimdi görsel test (test_sesi.wav dosyan klasörde olmalı)
    # Eğer dosya adın farklıysa burayı düzelt (Örn: 'Test.wav')
    engine.sesi_gorsellestir("Test2.wav")

