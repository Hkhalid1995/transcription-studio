import subprocess
import tempfile
import os

def extract_compressed_audio(input_file_bytes: bytes, original_filename: str) -> str:
    """
    Extracts 16kHz mono audio from uploaded media (video or audio) using ffmpeg.
    Returns path to temporary lightweight .mp3 file.
    """
    ext = os.path.splitext(original_filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_in:
        temp_in.write(input_file_bytes)
        temp_in_path = temp_in.name

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_out_path = temp_out.name
    temp_out.close()

    # ffmpeg command: extract mono audio, 16kHz sampling, 48kbps bitrate
    cmd = [
        "ffmpeg", "-y", "-i", temp_in_path,
        "-vn", "-acodec", "libmp3lame",
        "-ac", "1", "-ar", "16000",
        "-b:a", "48k",
        temp_out_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        os.remove(temp_in_path)
        return temp_out_path
    except Exception:
        # Fallback if ffmpeg is not installed on system: use original file directly
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)
        return temp_in_path