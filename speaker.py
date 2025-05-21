# # speaker.py
# from gtts import gTTS
# import pygame
# import tempfile
# import threading
#
# import tempfile
#
# def speak(text, lang='ru'):
#     tts = gTTS(text=text, lang=lang)
#     with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3', dir="C:\\Users\\mypol\\OneDrive\\Desktop\\BOTLESS") as fp:
#         tts.save(fp.name)
#         pygame.mixer.init()
#         pygame.mixer.music.load(fp.name)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)
#
# def async_speak(text, lang='ru'):
#     """Функция для асинхронного запуска голосовых сообщений."""
#     thread = threading.Thread(target=speak, args=(text,), kwargs={'lang': lang})
#     thread.start()
# import tempfile
# from gtts import gTTS
# import pygame
# import threading
# import os
#
#
# def speak(text, lang='ru'):
#     # Create gTTS object
#     tts = gTTS(text=text, lang=lang)
#     # Save the audio file to the system temporary directory
#     temp_dir = tempfile.gettempdir()
#     temp_file_path = os.path.join(temp_dir, 'temp_speech.mp3')
#     tts.save(temp_file_path)
#
#     # Initialize and play audio with pygame
#     pygame.mixer.init()
#     pygame.mixer.music.load(temp_file_path)
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():  # Wait until the file has finished playing
#         pygame.time.Clock().tick(10)
#
#     # Clean up the temporary file after playback
#     os.remove(temp_file_path)
#
#
# def async_speak(text, lang='ru'):
#     """Run speak function asynchronously."""
#     thread = threading.Thread(target=speak, args=(text,), kwargs={'lang': lang})
#     thread.start()
# import tempfile
# from gtts import gTTS
# import pygame
# import os
#
#
# def async_speak(text, lang='ru'):
#     # Create gTTS object
#     tts = gTTS(text=text, lang=lang)
#
#     # Save the audio file to the system temporary directory
#     temp_dir = tempfile.gettempdir()
#     temp_file_path = os.path.join(temp_dir, 'temp_speech.mp3')
#     tts.save(temp_file_path)
#
#     # Initialize and play audio with pygame
#     pygame.mixer.init()
#     pygame.mixer.music.load(temp_file_path)
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():  # Wait until the file has finished playing
#         pygame.time.Clock().tick(10)
#
#     # Clean up the temporary file after playback
#     os.remove(temp_file_path)
import pygame

def speak(audio_file):
    """Plays the specified audio file."""
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait until the audio finishes playing
        pygame.time.Clock().tick(10)

# Example usage based on prediction outcome
def handle_prediction(good_prob):
    if good_prob > 60:
        speak("goodENG.mp3")  # Plays "Спасибо, это пластиковая бутылка"
    elif good_prob < 50:
        speak("bad.mp3")  # Plays "Это не пластиковая бутылка"
