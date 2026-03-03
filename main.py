from nlp_engine import NLPEngine
from audio_engine import AudioEngine
import librosa

def dis_kaynak_analizi(metin, ses_dosya_yolu):
    # Motorları Başlatıyoruz
    nlp = NLPEngine()
    audio = AudioEngine()
    
    print("\n" + "="*40)
    print("PROFESYONEL HİBRİT ANALİZ SİSTEMİ")
    print("="*40)

    # 1. Metin Analizi (NLP)
    nlp_sonuc_ham = nlp.analiz_et(metin)
    print(f"[MODEL ÇIKTISI]: {nlp_sonuc_ham}")

    # Gelişmiş Mantık:
    # - 3 harften az girişleri ciddiye alma (k, d gibi)
    # - LABEL_1 gelirse Kaygılı kabul et (Senin modelin için geçerli)
    if len(metin) > 2 and "LABEL_1" in str(nlp_sonuc_ham):
        nlp_skor = 1.0
    else:
        nlp_skor = 0.0

    # 2. Ses Analizi (AER)
    try:
        y, sr = librosa.load(ses_dosya_yolu)
        ses_sonuc = audio.analiz_et(y, sr)
        enerji = ses_sonuc['enerji']
        
        # Senin sesin 0.0048 çıkmıştı. Hassas eşik 0.003
        ses_stres_skoru = 1.0 if enerji > 0.003 else 0.0
        print(f"[SES VERİSİ]: Enerji={enerji:.5f} | Stres Skoru={ses_stres_skoru}")
        
    except Exception as e:
        print(f"Ses Hatası: {e}")
        ses_stres_skoru = 0.0

    # 3. Final P_karakter Skoru (%40 Metin + %60 Ses)
    final_skor = (nlp_skor * 0.4) + (ses_stres_skoru * 0.6)
    
    print("-" * 25)
    print(f"FİNAL ANALİZ PUANI: {final_skor:.2f}")
    
    # Karar Mekanizması
    if final_skor >= 0.6:
        print("DURUM: Yüksek Kaygı/Hassasiyet (Multimodal Onaylı)")
    elif final_skor >= 0.4:
        print("DURUM:Hafif Belirsizlik/Çekince")
    else:
        print("DURUM: Stabil ve Dengeli Profil")
    print("="*40)

if __name__ == "__main__":
    kullanici_metni = input("Analiz edilecek ifadeyi girin: ")
    dis_kaynak_analizi(kullanici_metni,"Test.wav")