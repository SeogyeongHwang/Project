import time
import requests
import speech_recognition as sr

SERVER_URL = "http://127.0.0.1:5000/command"

def main():
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Calibrating mic noise... (1 sec)")
        r.adjust_for_ambient_noise(source, duration=1)
        
        print("listening. Say: 'turn on radio', 'stop radio', 'increase volume', 'reduce volume'")
        while True:
            try:
                audio = r.listen(source, timeout=None, phrase_time_limit=4)
                
                text = r.recognize_google(audio).lower()
                print("heard:", text)
                
                resp = requests.post(SERVER_URL, json={"text": text}, timeout=3)
                print("Server:", resp.json())
                
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand you.")
                requests.post(SERVER_URL, json={"text": ""}, timeout=3)
    
            except sr.RequestError as e:
                print("Speech API error:", e)
                
            except KeyboardInterrupt:
                print("\nStopping.")
                break
            
            time.sleep(0.1)
            
if __name__ == "__main__":
    main()