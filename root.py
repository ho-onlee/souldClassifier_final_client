import src.record_audio as record_audio
import src.decibel_meter as decibel_meter

def callback(frames):
    print(f"\nSaved {len(frames)} frames to WAV file.")
    print(f"\ndb Level: {meter.read_decibel()} dB SPL")

if __name__ == "__main__":
    recorder = record_audio.AudioRecorder(
        output_path="Recordings",
        duration_seconds=10,
        source_index="2"
    )
    meter = decibel_meter.DecibelMeter()
    recorder.setCallback(callback)
    recorder.start_stream()