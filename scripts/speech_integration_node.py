#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import subprocess
import threading
import os
import hashlib
import tempfile

from std_msgs.msg import String, Bool
from movement_api.msg import NavStatus


class SpeechIntegrationNode:
    def __init__(self):
        rospy.init_node("speech_integration_node")

        # --------------------
        # 1. Konfiguration & Pfade
        # --------------------
        self.piper_path = rospy.get_param("~piper_path", "/home/ubuntu/speech_out/piper/piper")
        self.model_path = rospy.get_param("~model_path", "/home/ubuntu/speech_out/piper_voices/en_ryan.onnx")
        self.sox_path   = rospy.get_param("~sox_path",   "/usr/bin/sox")
        self.aplay_path = rospy.get_param("~aplay_path", "/usr/bin/aplay")
        self.device = rospy.get_param("~device", "plughw:2,0")
        self.music_file = rospy.get_param("~music_file", "/home/ubuntu/music/musik/musik1_48k.wav")

        # WAV Cache
        self.enable_cache = rospy.get_param("~enable_cache", True)
        self.cache_dir = rospy.get_param("~cache_dir", "/home/ubuntu/speech_out/cache_wav")
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            rospy.logerr(f"[SpeechIntegration] Konnte cache_dir nicht erstellen: {self.cache_dir} ({e})")
            self.enable_cache = False

        # --------------------
        # 2. Topics (Schnittstellen zu den Teams)
        # --------------------
        # Textquelle (POI / Directions) – Topic-Name ist per Param konfigurierbar
        self.text_topic = rospy.get_param("~text_topic", "/speech_out/say")

        # TEAM MOVEMENT
        self.nav_status_topic  = rospy.get_param("~nav_status_topic", "/navbot/nav_status")
        self.speech_done_topic = rospy.get_param("~speech_done_topic", "/navbot/speech_done")

        # Debugging / Handshake (z.B. Directions)
        self.is_speaking_topic = rospy.get_param("~is_speaking_topic", "/speech_out/is_speaking")

        # --------------------
        # 3. State Management
        # --------------------
        self._lock = threading.Lock()
        self._speech_thread = None

        self.current_text = "Target reached. No text received."
        self.is_speaking = False
        self.music_process = None
        self.last_nav_state = None

        # --------------------
        # 4. Publisher & Subscriber
        # --------------------
        self.is_speaking_pub = rospy.Publisher(self.is_speaking_topic, Bool, queue_size=1, latch=True)
        self.speech_done_pub = rospy.Publisher(self.speech_done_topic, Bool, queue_size=1)

        rospy.Subscriber(self.text_topic, String, self._on_text_in, queue_size=10)
        rospy.Subscriber(self.nav_status_topic, NavStatus, self._on_nav_status_from_movement, queue_size=10)

        self.is_speaking_pub.publish(Bool(data=False))

        rospy.loginfo("==============================================")
        rospy.loginfo("[SpeechIntegration] NODE GESTARTET.")
        rospy.loginfo(f"Höre auf Text:   {self.text_topic}")
        rospy.loginfo(f"Höre auf Status: {self.nav_status_topic}")
        rospy.loginfo(f"Cache enabled:   {self.enable_cache}")
        rospy.loginfo(f"Cache dir:       {self.cache_dir}")
        rospy.loginfo("==============================================")

        rospy.on_shutdown(self._on_shutdown)

    # --------------------
    # Text Input Callback
    # --------------------
    def _on_text_in(self, msg: String):
        raw_text = (msg.data or "").strip()
        if raw_text:
            self.current_text = raw_text
            rospy.loginfo(f"[SpeechIntegration] Text gepuffert: '{self.current_text[:60]}...'")

    # --------------------
    # Movement Callback
    # --------------------
    def _on_nav_status_from_movement(self, msg: NavStatus):
        self.last_nav_state = msg.state

        if msg.state == NavStatus.MOVING_TO_TARGET:
            self._start_music()

        elif msg.state == NavStatus.ARRIVED:
            self._stop_music()
            self._start_speech_async()

        elif msg.state in (NavStatus.EMERGENCY_STOP, NavStatus.FAILED):
            self._stop_music()
            # kein speech_done, weil Ablauf unterbrochen

    # --------------------
    # Musik Logik
    # --------------------
    def _start_music(self):
        if self.music_process is not None:
            return

        try:
            rospy.loginfo(f"[SpeechIntegration] Starte Musik auf Device: {self.device}")
            rospy.loginfo(f"[SpeechIntegration] Datei: {self.music_file}")
            self.music_process = subprocess.Popen([self.aplay_path, "-D", self.device, self.music_file])
        except Exception as e:
            rospy.logerr(f"[SpeechIntegration] Fehler beim Musikstart: {e}")

    def _stop_music(self):
        if self.music_process is None:
            return

        rospy.loginfo("[SpeechIntegration] Stoppe Musik.")
        try:
            self.music_process.terminate()
            self.music_process.wait(timeout=0.5)
        except Exception:
            pass
        self.music_process = None

    # --------------------
    # Speech Threading
    # --------------------
    def _start_speech_async(self):
        with self._lock:
            if self._speech_thread is not None and self._speech_thread.is_alive():
                rospy.logwarn("[SpeechIntegration] Ich spreche bereits! Ignoriere neuen Trigger.")
                return
            self._speech_thread = threading.Thread(target=self._do_speech_cycle, daemon=True)
            self._speech_thread.start()

    # --------------------
    # Cache Helpers
    # --------------------
    def _normalize_text(self, text: str) -> str:
        # Stabilisiert Cache-Key bei Mehrfachspaces/Zeilenumbrüchen
        return " ".join((text or "").strip().split())

    def _cache_key_md5(self, text: str) -> str:
        # Modell/Voice in den Key aufnehmen, damit verschiedene Stimmen nicht kollidieren
        material = f"{self._normalize_text(text)}|model={self.model_path}"
        return hashlib.md5(material.encode("utf-8")).hexdigest()

    def _cache_path_for_text(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{self._cache_key_md5(text)}.wav")

    def _play_wav_file(self, wav_path: str) -> bool:
        try:
            p = subprocess.Popen([self.aplay_path, "-D", self.device, wav_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            p.wait()
            if p.returncode != 0:
                rospy.logerr(f"[SpeechIntegration] aplay Fehler: {p.stderr.read().decode(errors='ignore')}")
                return False
            return True
        except Exception as e:
            rospy.logerr(f"[SpeechIntegration] WAV Playback Exception: {e}")
            return False

    def _generate_wav_to_cache(self, text: str, out_wav_path: str) -> bool:
        """
        Piper -> SoX schreibt WAV in temp-Datei, dann atomar nach out_wav_path.
        SoX sorgt hier für 48kHz + Stereo wie in eurer bisherigen Pipeline.
        """
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="tts_", suffix=".wav", dir=self.cache_dir)
            os.close(fd)

            # 1) Piper (stdout WAV)
            p1 = subprocess.Popen(
                [self.piper_path, "--model", self.model_path, "--output_file", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # 2) SoX (stdin WAV -> tmp_path WAV, 48k stereo)
            p2 = subprocess.Popen(
                [self.sox_path, "-t", "wav", "-", "-r", "48000", "-c", "2", "-t", "wav", tmp_path],
                stdin=p1.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            p1.stdout.close()

            # Text rein
            p1.stdin.write((text + "\n").encode("utf-8"))
            p1.stdin.close()

            p2.wait()
            p1.wait()

            if p2.returncode != 0:
                rospy.logerr(f"[SpeechIntegration] SoX Fehler: {p2.stderr.read().decode(errors='ignore')}")
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return False

            # Atomar ersetzen
            os.replace(tmp_path, out_wav_path)
            return True

        except Exception as e:
            rospy.logerr(f"[SpeechIntegration] Cache-Generierung Exception: {e}")
            return False

    # --------------------
    # Speech Cycle
    # --------------------
    def _do_speech_cycle(self):
        text = self._normalize_text(self.current_text)
        self._set_speaking(True)

        try:
            rospy.loginfo(f"[SpeechIntegration] Spreche jetzt: {text}")

            # Cache nutzen
            if self.enable_cache and text:
                cached_path = self._cache_path_for_text(text)

                if os.path.exists(cached_path):
                    rospy.loginfo(f"[SpeechIntegration] Cache HIT: {cached_path}")
                    ok = self._play_wav_file(cached_path)
                else:
                    rospy.loginfo(f"[SpeechIntegration] Cache MISS: Erzeuge {cached_path}")
                    if self._generate_wav_to_cache(text, cached_path):
                        ok = self._play_wav_file(cached_path)
                    else:
                        ok = False

                if not ok:
                    rospy.logwarn("[SpeechIntegration] Cache-Pfad fehlgeschlagen, fallback auf Stream-Pipeline.")
                    self._play_stream_pipeline(text)

            else:
                # Cache aus: direkt Stream-Pipeline
                self._play_stream_pipeline(text)

        finally:
            rospy.loginfo("[SpeechIntegration] Fertig. Sende Signal an Movement.")
            self.speech_done_pub.publish(Bool(data=True))
            self._set_speaking(False)

            if self.last_nav_state == NavStatus.MOVING_TO_TARGET:
                self._start_music()

    def _play_stream_pipeline(self, text: str) -> bool:
        """Euer alter Weg: Piper -> SoX -> aplay als Stream."""
        try:
            p1 = subprocess.Popen(
                [self.piper_path, "--model", self.model_path, "--output_file", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            p2 = subprocess.Popen(
                [self.sox_path, "-t", "wav", "-", "-r", "48000", "-c", "2", "-t", "wav", "-"],
                stdin=p1.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p1.stdout.close()

            p3 = subprocess.Popen(
                [self.aplay_path, "-D", self.device],
                stdin=p2.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            p2.stdout.close()

            p1.stdin.write((text + "\n").encode("utf-8"))
            p1.stdin.close()

            p3.wait()
            p2.wait()
            p1.wait()

            if p3.returncode != 0:
                rospy.logerr(f"[SpeechIntegration] Audio Fehler: {p3.stderr.read().decode(errors='ignore')}")
                return False
            return True

        except Exception as e:
            rospy.logerr(f"[SpeechIntegration] Stream-Pipeline Exception: {e}")
            return False

    def _set_speaking(self, active: bool):
        self.is_speaking = active
        self.is_speaking_pub.publish(Bool(data=active))

    def _on_shutdown(self):
        self._stop_music()


if __name__ == "__main__":
    try:
        SpeechIntegrationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
