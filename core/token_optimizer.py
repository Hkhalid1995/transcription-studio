import json
import pandas as pd
from typing import Dict, Any

class TokenOptimizer:
    @staticmethod
    def cues_to_compact_tsv(df: pd.DataFrame) -> str:
        """Converts cue boundaries into a minimal pipe-delimited string."""
        clean_df = df.copy()
        clean_df.columns = [c.strip() for c in clean_df.columns]
        
        id_col = next((c for c in clean_df.columns if "id" in c.lower() or "cue" in c.lower()), clean_df.columns[0])
        start_col = next((c for c in clean_df.columns if "start" in c.lower()), clean_df.columns[1])
        end_col = next((c for c in clean_df.columns if "end" in c.lower()), clean_df.columns[2])
        
        mini_df = clean_df[[id_col, start_col, end_col]].rename(
            columns={id_col: "cue_id", start_col: "start", end_col: "end"}
        )
        return mini_df.to_csv(sep="|", index=False)

    @staticmethod
    def delivered_to_compact_transcription_table(df: pd.DataFrame) -> str:
        """
        Pulls ONLY transcription, speaker, and emotion columns.
        Drops translation columns to save ~40-50% in Pass 2 input tokens.
        """
        mapping = {}
        for col in df.columns:
            c = col.lower()
            if "id" in c or "cue" in c:
                mapping[col] = "id"
            elif "start" in c:
                mapping[col] = "start"
            elif "end" in c:
                mapping[col] = "end"
            elif "transcription" in c:
                mapping[col] = "tr"
            elif "speaker" in c:
                mapping[col] = "spk"
            elif "emotion" in c:
                mapping[col] = "emo"

        compact_df = df.rename(columns=mapping)
        target_cols = [v for v in ["id", "start", "end", "tr", "spk", "emo"] if v in compact_df.columns]
        return compact_df[target_cols].to_csv(sep="|", index=False)

    @staticmethod
    def compact_pass1_for_transcription_qa(pass1_data: Dict[str, Any]) -> str:
        """
        Minifies Pass 1 payload focusing strictly on audio verbatim truth,
        phonetics, and dialect items.
        """
        optimized_cues = []
        for cue in pass1_data.get("cues", []):
            cue_compact = {
                "id": str(cue.get("cue_id", "")),
                "start": cue.get("start", ""),
                "end": cue.get("end", ""),
                "verbatim": cue.get("verbatim", ""),
                "gloss": cue.get("literal_gloss", ""),
                "qual": cue.get("audio_quality", "clean"),
                "boundary": cue.get("boundary", "contained"),
                "gender": cue.get("voice_gender", ""),
                "emo": cue.get("delivered_emotion", ""),
                "conf": cue.get("confidence", 1.0)
            }
            if cue.get("dialect_sensitive_items"):
                cue_compact["dialect_items"] = cue["dialect_sensitive_items"]
            if cue.get("entities"):
                cue_compact["entities"] = cue["entities"]
            optimized_cues.append(cue_compact)

        return json.dumps({
            "cues": optimized_cues,
            "uncovered": pass1_data.get("uncovered_speech", [])
        }, ensure_ascii=False, separators=(',', ':'))