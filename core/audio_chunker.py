import subprocess
import os
import tempfile
import pandas as pd
from typing import List, Dict, Any

def parse_timecode_to_seconds(tc: str, fps: float = 24.0) -> float:
    """Converts HH:MM:SS:FF or HH:MM:SS.mmm to seconds."""
    parts = str(tc).strip().replace(",", ".").split(":")
    if len(parts) == 4:
        h, m, s, f = map(float, parts)
        return h * 3600 + m * 60 + s + (f / fps)
    elif len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    return 0.0

def format_seconds_to_tc(seconds: float, fps: float = 24.0) -> str:
    """Converts seconds to HH:MM:SS:FF timecode."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int(round((seconds - int(seconds)) * fps))
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

def get_audio_duration(audio_path: str) -> float:
    """Gets total audio duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0

def slice_audio_segment(audio_path: str, start_sec: float, duration_sec: float) -> str:
    """Slices a segment of audio without re-encoding."""
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out_path = out_file.name
    out_file.close()

    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", audio_path,
        "-acodec", "copy",
        out_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return out_path
    except Exception:
        # Fallback to returning original path if slicing fails
        return audio_path

def create_contiguous_chunks(
    df: pd.DataFrame,
    audio_path: str,
    chunk_duration_minutes: int = 7,
    fps: float = 24.0
) -> List[Dict[str, Any]]:
    """Splits audio & cues into contiguous chunks so no dialogue is lost."""
    total_sec = get_audio_duration(audio_path)
    chunk_sec = chunk_duration_minutes * 60.0

    start_col = next((c for c in df.columns if "start" in c.lower()), df.columns[1])
    end_col = next((c for c in df.columns if "end" in c.lower()), df.columns[2])

    if total_sec <= 0:
        last_tc = df[end_col].iloc[-1]
        total_sec = parse_timecode_to_seconds(last_tc, fps) + 5.0

    chunks = []
    current_start = 0.0
    chunk_idx = 1

    while current_start < total_sec:
        current_end = min(current_start + chunk_sec, total_sec)
        
        # Filter cues whose start time falls within this chunk
        chunk_cues = df[df[start_col].apply(
            lambda tc: current_start <= parse_timecode_to_seconds(tc, fps) < current_end
        )]
        
        chunks.append({
            "chunk_index": chunk_idx,
            "start_sec": current_start,
            "end_sec": current_end,
            "duration_sec": current_end - current_start,
            "start_tc": format_seconds_to_tc(current_start, fps),
            "end_tc": format_seconds_to_tc(current_end, fps),
            "cues_df": chunk_cues
        })
        
        current_start = current_end
        chunk_idx += 1

    return chunks