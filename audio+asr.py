import whisper
import librosa
import noisereduce as nr
import soundfile as sf

#1 audio downloading
audio_path = r"G:\DEPI\AI\Project\audio3.wav"
audio, sr = librosa.load(audio_path, sr=None)

#2  noise reduction
reduced_noise = nr.reduce_noise(y=audio, sr=sr)

#3 after noise reduction
clean_path = "clean.wav"
sf.write(clean_path, reduced_noise, sr)

#4 asr
model = whisper.load_model("base")
result = model.transcribe(clean_path, language="en")

#5 result
print(result["text"])
