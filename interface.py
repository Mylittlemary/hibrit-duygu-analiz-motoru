import tkinter as tk
from tkinter import ttk
from nlp_engine import NLPEngine
from audio_engine import AudioEngine
import librosa
import sounddevice as sd # Yeni eklenen
from scipy.io import wavfile # Yeni eklenen
import numpy as np

class AnalizArayuzu:
    def __init__(self, root):
        self.root = root
        self.root.title("Mühendis - Hibrit Duygu Analiz Motoru v1.0")
        self.root.geometry("500x700") # Boyutu biraz büyüttük
        
        self.nlp = NLPEngine()
        self.audio = AudioEngine()
        
        self.arayuz_olustur()

    def arayuz_olustur(self):
        # 1. Metin Girişi
        tk.Label(self.root, text="Analiz Edilecek Metin:", font=("Arial", 10, "bold")).pack(pady=10)
        self.metin_giris = tk.Entry(self.root, width=50)
        self.metin_giris.pack(pady=5)
        self.metin_giris.insert(0, "çok mutsuzum")

        # 2. Hassasiyet Ayarı
        tk.Label(self.root, text="Ses Hassasiyet Eşiği (Threshold):", font=("Arial", 10, "bold")).pack(pady=10)
        self.esik_slider = tk.Scale(self.root, from_=0.0001, to=0.05, resolution=0.0001, orient="horizontal", length=300)
        self.esik_slider.set(0.0092) 
        self.esik_slider.pack()

        # --- YENİ: SES KAYIT BUTONU BURAYA EKLENDİ ---
        tk.Label(self.root, text="Ses Analizi İçin:", font=("Arial", 10, "bold")).pack(pady=10)
        self.btn_kayit = tk.Button(self.root, text="ŞİMDİ SESİMİ KAYDET (3 SN)", command=self.ses_kaydet, bg="orange", font=("Arial", 10, "bold"))
        self.btn_kayit.pack(pady=5)
        # --------------------------------------------

        # 3. Analiz Butonu
        self.btn_analiz = tk.Button(self.root, text="HİBRİT ANALİZİ BAŞLAT", command=self.analiz_yap, bg="green", fg="white", font=("Arial", 12, "bold"))
        self.btn_analiz.pack(pady=20)

        # 4. Sonuç Paneli
        self.sonuc_frame = tk.LabelFrame(self.root, text=" Analiz Sonuçları ", padx=10, pady=10)
        self.sonuc_frame.pack(pady=10, fill="both", expand=True)

        self.lbl_nlp = tk.Label(self.sonuc_frame, text="Metin Skoru: -")
        self.lbl_nlp.pack(anchor="w")
        
        self.lbl_ses = tk.Label(self.sonuc_frame, text="Ses Enerjisi: -")
        self.lbl_ses.pack(anchor="w")

        self.lbl_final = tk.Label(self.sonuc_frame, text="FİNAL PUANI: -", font=("Arial", 14, "bold"))
        self.lbl_final.pack(pady=10)

        self.lbl_durum = tk.Label(self.sonuc_frame, text="DURUM: Bekleniyor...", fg="blue")
        self.lbl_durum.pack()

    # --- YENİ: SES KAYDETME FONKSİYONU BURAYA EKLENDİ ---
    def ses_kaydet(self):
        fs = 44100  
        saniye = 3  
        self.lbl_durum.config(text="DURUM: Kayıt yapılıyor (3 sn)... Mikrofonu kullanın!", fg="red")
        self.root.update()
        
        try:
            kayit = sd.rec(int(saniye * fs), samplerate=fs, channels=1)
            sd.wait()  
            
            # Kayıt sonrası normalize etme (daha iyi sonuç için)
            kayit_scaled = np.int16(kayit * 32767)
            wavfile.write("Test2.wav", fs, kayit_scaled)
            
            self.lbl_durum.config(text="DURUM: Kayıt Test2.wav olarak kaydedildi.", fg="green")
        except Exception as e:
            self.lbl_durum.config(text=f"Kayıt Hatası: {e}", fg="red")

    def analiz_yap(self):
        metin = self.metin_giris.get()
        esik_degeri = self.esik_slider.get()
        
        try:
            # 1. NLP Analizi
            nlp_res = self.nlp.analiz_et(metin)
            res_str = str(nlp_res).lower()
            
            if "positive" in res_str or "label_1" in res_str:
                nlp_skor = 0.0
                model_label = "OLUMLU / STABİL"
            else:
                nlp_skor = 1.0
                model_label = "OLUMSUZ / KAYGILI"

            # 2. Ses Analizi
            y, sr = librosa.load("Test2.wav") 
            ses_res = self.audio.analiz_et(y, sr)
            enerji = ses_res['enerji']
            
            ses_skor = 1.0 if enerji > esik_degeri else 0.0
            
            # 3. Hibrit Puan
            final_skor = (nlp_skor * 0.4) + (ses_skor * 0.6)
            
            # 4. Arayüz Güncelleme
            self.lbl_nlp.config(text=f"Metin Skoru: {nlp_skor} ({model_label})")
            self.lbl_ses.config(text=f"Ses Enerjisi: {enerji:.5f} (Eşik: {esik_degeri:.4f})")
            self.lbl_final.config(text=f"FİNAL PUANI: {final_skor:.2f}")
            
            if final_skor >= 0.6:
                self.lbl_durum.config(text="DURUM: YÜKSEK KAYGI", fg="red")
            elif final_skor >= 0.4:
                self.lbl_durum.config(text="DURUM: HAFİF ÇEKİNCE", fg="orange")
            else:
                self.lbl_durum.config(text="DURUM: STABİL", fg="green")
                
        except Exception as e:
            self.lbl_durum.config(text=f"Analiz Hatası: {e}", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalizArayuzu(root)
    root.mainloop()

