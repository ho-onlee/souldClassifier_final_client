import argparse
import sys
from multiprocessing import Process
import wave
import time, datetime
import os
import sounddevice as sd


class AudioRecorder:
    def __init__(
        self,
        output_path: str,
        duration_seconds: int = 10,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_size: int = 1024,
        source_index: int = None
    ):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.source_index = source_index
        self.animationframes = [
            "   ..   ",
            "   ()   ",
            "  (())  ",
            " ((  )) ",
            "((    ))",
            "(      )",
        ]

    def save_recording(self, frames: list) -> None:
        """Save the recorded audio frames to a WAV file."""
        name = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with wave.open(os.path.join(self.output_path, name), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(frames))
        sys.stdout.write("\r                                                          ")
        sys.stdout.flush()
        sys.stdout.write(f"Saved recording to {os.path.join(self.output_path, name)}")
        sys.stdout.flush()

    

    def start_animation(self) -> None:
        """Start a simple animation in the console."""
        animation_frame = 0
        while True:
            sys.stdout.write(f"\r{self.animationframes[animation_frame % len(self.animationframes)]}")
            sys.stdout.flush()
            animation_frame += 1
            time.sleep(0.1)
            if animation_frame >= len(self.animationframes):  # Loop through the animation frames twice
                animation_frame = 0
            
    def save_process(self, args):
        save_recording_process = Process(target=self.save_recording, args=args, daemon=True)
        save_recording_process.start()
        save_recording_process.join()

        
    def record_audio(self) -> None:
        """Record audio from the default input device and save it as a WAV file."""
        stream = None
        frame1_buffer = []
        frame2_buffer = []
        animation_process = Process(target=self.start_animation, daemon=True)
        

        try:
            stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                device=self.source_index,
                dtype="int16",
                blocksize=self.chunk_size,
            )
            stream.start()

            should_frame2_record = False
            total_chunks = int(self.sample_rate / self.chunk_size * self.duration_seconds)
            print("Recording audio... Press Ctrl+C to stop.")
            
            animation_process.start()
            while True:
                frame, _ = stream.read(self.chunk_size)
                frame = frame.tobytes()

                frame1_buffer.append(frame)
                if len(frame1_buffer) >= total_chunks // 2 and not should_frame2_record:
                    should_frame2_record = True                    
                if should_frame2_record:
                    frame2_buffer.append(frame)
                if len(frame1_buffer) >= total_chunks:
                    self.save_process((frame1_buffer,))
                    frame1_buffer = []
                if len(frame2_buffer) >= total_chunks:
                    self.save_process((frame2_buffer,))
                    frame2_buffer = []
                
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
            if frame1_buffer:                
                self.save_process((frame1_buffer,))
            if frame2_buffer:
                self.save_process((frame2_buffer,))
        finally:
            print(len(frame1_buffer), len(frame2_buffer))
            if stream is not None:
                stream.stop()
                stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record 10 seconds of audio and save as WAV")
    parser.add_argument(
        "-o",
        "--output",
        default="recorded_audio.wav",
        help="Output WAV file path (default: recorded_audio.wav)",
    )
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        help="Input audio device index (default: default input device)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=10,
        help="Recording duration in seconds (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args._get_args())
    if args.source is not None:
        print(f"Using audio input device index: {args.source}")
    else:
        devices = sd.query_devices()
        for i, info in enumerate(devices):
            print(f"[{i}] {info['name']}  (inputs={info['max_input_channels']}, outputs={info['max_output_channels']})")
        args.source = int(input("Enter the audio input device index to use: "))

        
    recorder = AudioRecorder(
        output_path=args.output,
        duration_seconds=args.duration,
        source_index=args.source
    )
    recorder.record_audio()