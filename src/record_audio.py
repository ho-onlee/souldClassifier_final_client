import argparse
import sys
from multiprocessing import Event, Process
import wave
import time, datetime
import os
import sounddevice as sd
from typing import Callable, Optional


def _animation_worker(animationframes: list[str], stop_event: Event) -> None:
    """Render a simple console animation in a separate process."""
    animation_frame = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{animationframes[animation_frame % len(animationframes)]}")
        sys.stdout.flush()
        animation_frame += 1
        time.sleep(0.1)


def _save_recording_worker(output_path: str, channels: int, sample_rate: int, frames: list) -> None:
    """Persist recorded frames to a WAV file in a separate process."""
    name = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    output_file = os.path.join(output_path, name)
    with wave.open(output_file, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))


class AudioRecorder:
    def __init__(
        self,
        output_path: Optional[str] = None,
        duration_seconds: int = 10,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_size: int = 1024,
        source_index: str = "",
        animation: bool = True,
    ):
        self.output_path = output_path
        if not os.path.exists(self.output_path) and self.output_path is not None:
            os.makedirs(self.output_path)
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.animation = animation
        self.source_index = int(source_index) if source_index.isdigit() else source_index
        self.stream = None
        self.callback = None
        self.animation_process = None
        self.stop_animation_event = Event()
        self.animationframes = [
            "   .LISTENING.   ",
            "   (LISTENING)   ",
            "  ((LISTENING))  ",
            " (( LISTENING )) ",
            "((  LISTENING  ))",
            "(   LISTENING   )",
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

    def setCallback(self, callback: Callable[[bytes], None]) -> None:
        """Set a callback function to be called with each recorded audio chunk."""
        self.callback = callback

    def start_animation(self) -> None:
        """Start a simple animation in the console."""
        if self.animation_process is None or not self.animation_process.is_alive():
            self.stop_animation_event.clear()
            self.animation_process = Process(
                target=_animation_worker,
                args=(self.animationframes, self.stop_animation_event),
                daemon=True,
            )
            self.animation_process.start()
            
    def save_process(self, args):
        if self.output_path is not None:
            save_recording_process = Process(
                target=_save_recording_worker,
                args=(self.output_path, self.channels, self.sample_rate, args[0]),
                daemon=True,
            )
            save_recording_process.start()
            save_recording_process.join()
        if self.callback is not None:
            self.callback(*args)

        
    def start_stream(self) -> None:
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            device=self.source_index,
            dtype="int16",
            blocksize=self.chunk_size,
        )
        self.stream.start()
        print("Recording Started")
        if self.animation:
            self.start_animation()

        should_frame2_record = False
        total_chunks = int(self.sample_rate / self.chunk_size * self.duration_seconds)
        print("Recording audio... Press Ctrl+C to stop.")
        frame_buffer = [[], []]

        try:
            while True:
                frame, _ = self.stream.read(self.chunk_size)
                frame = frame.tobytes()

                frame_buffer[0].append(frame)
                if len(frame_buffer[0]) >= total_chunks // 2 and not should_frame2_record:
                    should_frame2_record = True
                if should_frame2_record:
                    frame_buffer[1].append(frame)

                if len(frame_buffer[0]) >= total_chunks:
                    self.save_process((frame_buffer[0],))
                    frame_buffer[0] = []
                if len(frame_buffer[1]) >= total_chunks:
                    self.save_process((frame_buffer[1],))
                    frame_buffer[1] = []
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
            if frame_buffer[0]:
                self.save_process((frame_buffer[0],))
            if frame_buffer[1]:
                self.save_process((frame_buffer[1],))
        finally:
            self.terminate()
                
            
    def terminate(self) -> None:
        """Terminate the recording and animation processes."""
        self.stop_animation_event.set()
        if self.animation_process is not None and self.animation_process.is_alive():
            self.animation_process.join(timeout=1.0)
            if self.animation_process.is_alive():
                self.animation_process.terminate()
                self.animation_process.join()
        self.animation_process = None
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    



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
    if args.source is not None:
        print(f"Using audio input device index: {args.source}")
    else:
        devices = sd.query_devices()
        for i, info in enumerate(devices):
            print(f"[{i}] {info['name']}  (inputs={info['max_input_channels']}, outputs={info['max_output_channels']})")
        args.source = input("Enter the audio input device index to use: ")

        
    recorder = AudioRecorder(
        output_path=args.output,
        duration_seconds=args.duration,
        source_index=args.source
    )
    recorder.setCallback(lambda frames: print(f"\nSaved {len(frames)} frames to WAV file."))
    recorder.start_stream()