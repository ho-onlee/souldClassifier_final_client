import src.record_audio as record_audio
import src.decibel_meter as decibel_meter
import datetime, time
import os
from multiprocessing import Process

def callback(frames):
    print(f"\nSaved {len(frames)} frames to WAV file.")
    print(f"\ndb Level: {meter.read_decibel()} dB SPL")

def db_level(time_interval):
    db_levels = []
    while True:
        db_levels.append([datetime.datetime.now(), meter.read_decibel()])
        time.sleep(time_interval)

def start_db_monitoring(time_interval=0.01):
    db_process = Process(target=db_level, args=(time_interval,), daemon=True)
    db_process.start()

if __name__ == "__main__":
    recorder = record_audio.AudioRecorder(
        output_path="Recordings",
        duration_seconds=10,
        source_index="2"
    )
    meter = decibel_meter.DecibelMeter()
    meter.set_averaging_time_ms(125)
    recorder.setCallback(callback)
    recorder.start_stream()
    start_db_monitoring()