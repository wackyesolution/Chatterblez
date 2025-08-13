add another step after the text has been converted to audio that will use
except it will use wav instead of mp3

def remove_silence_pydub(input_file, output_file, silence_thresh=-40, min_silence_len=500):
    """
    Remove silences from audio file using pydub

    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        silence_thresh: Silence threshold in dBFS (lower = more sensitive)
        min_silence_len: Minimum silence length in milliseconds to remove
    """
    # Load audio file
    audio = AudioSegment.from_file(input_file)

    # Split audio on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,  # Min silence length in ms
        silence_thresh=silence_thresh,    # Silence threshold in dBFS
        keep_silence=100                  # Keep 100ms of silence for natural flow
    )

    # Combine chunks
    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk

    # Export result
    combined.export(output_file, format="mp3")
    print(f"Processed audio saved to: {output_file}")

    # Print statistics
    original_duration = len(audio) / 1000  # Convert to seconds
    new_duration = len(combined) / 1000
    removed_time = original_duration - new_duration

    print(f"Original duration: {original_duration:.2f}s")
    print(f"New duration: {new_duration:.2f}s")
    print(f"Removed silence: {removed_time:.2f}s ({removed_time/original_duration*100:.1f}%)")
    add a checkbox named "enable silence trimming" if disabled the below is also disabled and not sent to core.py
    add the  min_silence_len=min_silence_len,  # Min silence length in ms
        silence_thresh=silence_thresh,    # Silence threshold in dBFS
        keep_silence=100                  # K


    to the qsettings similar to model settings
    do not remove anything from settings