from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, ConfigDict, field_validator

class ResilientModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        coerce_numbers_to_str=False,
        arbitrary_types_allowed=True
    )

# --- PASS 1 SCHEMAS ---
class Pass1Dialect(ResilientModel):
    variety: str = ""
    markers: str = ""
    differs_from_expected: bool = False

class Pass1DialectSensitiveItem(ResilientModel):
    item: str = ""
    dialect_sense: str = ""
    standard_misreading: str = ""

class Pass1Cue(ResilientModel):
    cue_id: Union[str, int] = ""
    start: str = ""
    end: str = ""
    speech_present: bool = True
    verbatim: str = ""
    literal_gloss: str = ""
    natural_gloss: str = ""
    dialect: Optional[Union[Pass1Dialect, Dict[str, Any], str]] = None
    dialect_sensitive_items: List[Union[Pass1DialectSensitiveItem, Dict[str, Any], str]] = []
    entities: List[Union[Dict[str, Any], str]] = []
    voice_gender: str = "unclear"
    delivered_emotion: str = ""
    audio_quality: str = "clean"
    boundary: str = "contained"
    confidence: Union[float, int, str] = 1.0

    @field_validator("cue_id", mode="before")
    @classmethod
    def convert_cue_id_to_str(cls, v):
        return str(v) if v is not None else ""

class Pass1UncoveredSpeech(ResilientModel):
    approx_start: str = ""
    approx_end: str = ""
    kind: str = "dialogue"
    verbatim: str = ""
    literal_gloss: str = ""
    voice_gender: str = ""
    confidence: Union[float, int, str] = 1.0

class Pass1Output(ResilientModel):
    video_ref_id: str = ""
    cues: List[Pass1Cue] = []
    uncovered_speech: List[Pass1UncoveredSpeech] = []

# --- PASS 2 TRANSCRIPTION SCHEMAS ---
class TranscriptionFinding(ResilientModel):
    error_type: str = "TRANSCRIPTION_MISMATCH"
    severity: str = "MAJOR"
    explanation: str = ""
    audio_evidence: str = ""
    transcription_evidence: str = ""
    suggested_transcription_fix: str = ""
    confidence: Union[float, int, str] = 1.0

class FlaggedTranscriptionLine(ResilientModel):
    cue_id: Union[str, int] = ""
    start: str = ""
    end: str = ""
    speaker: str = ""
    status: str = "FLAGGED"
    audio_verbatim: str = ""
    audio_gloss: str = ""
    delivered_transcription: str = ""
    findings: List[TranscriptionFinding] = []

    @field_validator("cue_id", mode="before")
    @classmethod
    def convert_cue_id_to_str(cls, v):
        return str(v) if v is not None else ""

class SystemicTranscriptionIssue(ResilientModel):
    pattern: str = ""
    error_type: str = ""
    affected_cue_ids: List[Union[str, int]] = []
    recommendation: str = ""

class MissingDialogueFinding(ResilientModel):
    approx_start: str = ""
    approx_end: str = ""
    severity: str = "MAJOR"
    audio_verbatim: str = ""
    audio_gloss: str = ""
    explanation: str = ""
    suggested_transcription: str = ""

class Pass2TranscriptionSummary(ResilientModel):
    lines_audited: int = 0
    lines_clean: int = 0
    lines_unverifiable: int = 0
    counts_by_severity: Dict[str, int] = {}
    counts_by_error_type: Dict[str, int] = {}
    verdict: str = "PASS"
    verdict_reason: str = ""

class Pass2Output(ResilientModel):
    video_ref_id: str = ""
    summary: Pass2TranscriptionSummary = Pass2TranscriptionSummary()
    systemic_transcription_issues: List[SystemicTranscriptionIssue] = []
    missing_dialogue: List[MissingDialogueFinding] = []
    flagged_lines: List[FlaggedTranscriptionLine] = []