import src.record_audio as record_audio
import src.decibel_meter as decibel_meter
import datetime, time
import os
from multiprocessing import Process

def callback(frames):
    print(f"\nSaved {len(frames)} frames to WAV file.")

def db_level(time_interval):
    db_levels = []
    while True:
        db_levels.append([datetime.datetime.now(), meter.read_decibel()])
        time.sleep(time_interval)
        if len(db_levels) > 1000:
            with open(os.path.join("Recordings", f"db_levels_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"), "a") as f:
                for timestamp, db in db_levels:
                    f.write(f"{timestamp},{db}\n")
            db_levels.clear()


if __name__ == "__main__":
    recorder = record_audio.AudioRecorder(
        output_path="Recordings",
        duration_seconds=10,
        source_index="2"
    )
    meter = decibel_meter.DecibelMeter()
    meter.set_averaging_time_ms(125)
    recorder.setCallback(callback)
    audio_stream = Process(target=recorder.start_stream)
    audio_stream.start()
    db_process = Process(target=db_level, args=(0.01,))
    db_process.start()
    audio_stream.join()
    db_process.join()