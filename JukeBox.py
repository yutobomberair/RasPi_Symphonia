import pygame
import threading
import time

class JukeBox:
    def __init__(self, music_path="./sounds/BGM.mp3"):
        self.music_path = music_path
        self.playing = False  # Whether music is currently playing
        self.music_thread = None

    # This function plays music (executed in a separate thread)
    def _play_music(self):
        pygame.mixer.init()
        pygame.mixer.music.load(self.music_path)
        pygame.mixer.music.play()
        self.playing = True
        print("Music is playing... (Press Enter to stop)")

        # Loop while music is playing
        while self.playing and pygame.mixer.music.get_busy():
            time.sleep(0.1)

    # Main control function: starts music and waits for user to press Enter
    def play_with_interrupt(self):
        self.music_thread = threading.Thread(target=self._play_music)
        self.music_thread.start()  # Start music in a separate thread

        # Main thread waits for user input
        input(">> Press Enter to stop the music: ")
        self.stop_music()

    # Function to stop music playback
    def stop_music(self):
        if self.playing:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            self.playing = False
            print("Music stopped.")

if __name__ == "__main__":
    inst = JukeBox("./sounds/BGM.mp3")
    inst.play_with_interrupt()
