import src.record_audio as record_audio
import src.decibel_meter as decibel_meter

def callback(frames):
    print(f"\nSaved {len(frames)} frames to WAV file.")
    print(f"\ndb Level: {meter.read_decibel()} dB SPL")

def log_db_level():
    with open(f"db_levels_{record_audio.get_timestamp()}.txt", "w") as f:
        while True:
            db_level = meter.read_decibel()
            timestamp = record_audio.get_timestamp()
            f.write(f"{timestamp}: {db_level} dB SPL\n")
            f.flush()
    print(f"\ndb Level: {meter.read_decibel()} dB SPL")

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