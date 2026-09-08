import static_ffmpeg
import static_ffmpeg.run as _sf_run
_sf_run.LOCK_FILE = "/tmp/static_ffmpeg.lock"
static_ffmpeg.add_paths(download_dir="/tmp/static_ffmpeg")
import streamlit as st
import json
import os
import re
import hashlib
import pandas as pd
from typing import Any
from google import genai

# Resilient JSON repair import with built-in zero-dependency fallback
try:
    from json_repair import repair_json
except ImportError:
    def repair_json(json_str: str) -> str:
        if not json_str:
            return "{}"
        s = str(json_str).strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = re.sub(r",\s*([\]}])", r"\1", s)
        open_braces = s.count("{") - s.count("}")
        open_brackets = s.count("[") - s.count("]")
        s += "]" * max(0, open_brackets)
        s += "}" * max(0, open_braces)
        return s

from core.token_optimizer import TokenOptimizer
from core.audio_utils import extract_compressed_audio
from core.audio_chunker import create_contiguous_chunks, slice_audio_segment
from core.cost_tracker import UsageStats
from core.schemas import Pass1Output, Pass2Output
import prompts

# -----------------------------------------------------------------------------
# Configuration & UI Initialization
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Forensic Audio-Truth Transcription QA Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CACHE_DIR = ".qa_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Dynamic Model Fetcher
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def fetch_available_gemini_models(api_key: str) -> list:
    """Discovers available models from the user's Gemini API key."""
    if not api_key:
        return []
    try:
        client = genai.Client(api_key=api_key)
        discovered = []
        for m in client.models.list():
            name = getattr(m, "name", "")
            if name.startswith("models/"):
                name = name.replace("models/", "")
            if name and "gemini" in name.lower() and not name.endswith("-vision"):
                discovered.append(name)
        return sorted(list(set(discovered)))
    except Exception:
        return []

# -----------------------------------------------------------------------------
# Resilient Helpers & Safe API Callers
# -----------------------------------------------------------------------------
def safe_extract_text(res: Any) -> str:
    """Safely extracts text from Gemini API response objects."""
    if res is None:
        return ""
    
    # 1. Direct text property
    try:
        if hasattr(res, "text") and res.text is not None:
            return str(res.text)
    except Exception:
        pass

    # 2. Iterate through candidates and parts
    if hasattr(res, "candidates") and res.candidates:
        candidate = res.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(str(part.text))
            if text_parts:
                return "".join(text_parts)
        
        # Check if model was blocked by safety or finish reason
        if hasattr(candidate, "finish_reason"):
            finish_reason = str(candidate.finish_reason)
            if finish_reason not in ["STOP", "1", "None"]:
                raise ValueError(f"Model generation stopped with finish reason: {finish_reason}")

    return ""

def clean_and_parse_json(raw_text: Any) -> dict:
    """Strips markdown code fences and auto-repairs malformed/truncated JSON."""
    if raw_text is None or not str(raw_text).strip():
        raise ValueError("The model returned an empty text response. Please verify your selected Model ID and API Key.")
    
    cleaned = str(raw_text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    
    try:
        return json.loads(cleaned)
    except Exception:
        repaired_str = repair_json(cleaned)
        return json.loads(repaired_str)

def parse_and_validate_pass1(raw_text: str) -> dict:
    parsed = clean_and_parse_json(raw_text)
    try:
        validated = Pass1Output.model_validate(parsed)
        return validated.model_dump()
    except Exception as val_err:
        st.warning(f"⚠️ Pass 1 schema reconciled: {val_err}")
        return parsed

def parse_and_validate_pass2(raw_text: str) -> dict:
    parsed = clean_and_parse_json(raw_text)
    try:
        validated = Pass2Output.model_validate(parsed)
        return validated.model_dump()
    except Exception as val_err:
        st.warning(f"⚠️ Pass 2 schema reconciled: {val_err}")
        return parsed

def call_gemini_safe(client, model: str, contents: list, system_instruction: str, temperature: float = 0.0, response_mime_type: str = "application/json"):
    """3-tier fallback to handle any model endpoint restrictions smoothly."""
    # Tier 1: Standard call with system_instruction and JSON mode
    try:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": response_mime_type,
                "temperature": temperature
            }
        )
    except Exception as err1:
        err_str1 = str(err1)
        fallback_contents = []
        sys_prepended = False
        for item in contents:
            if isinstance(item, str) and not sys_prepended:
                fallback_contents.append(f"### SYSTEM INSTRUCTION:\n{system_instruction}\n\n### USER DATA:\n{item}")
                sys_prepended = True
            else:
                fallback_contents.append(item)
        if not sys_prepended:
            fallback_contents.append(f"### SYSTEM INSTRUCTION:\n{system_instruction}")

        # Tier 2: Try without system_instruction in config
        if "Developer instruction" in err_str1 or "system_instruction" in err_str1:
            try:
                return client.models.generate_content(
                    model=model,
                    contents=fallback_contents,
                    config={
                        "response_mime_type": response_mime_type,
                        "temperature": temperature
                    }
                )
            except Exception as err2:
                err_str2 = str(err2)
                if "JSON mode" not in err_str2 and "response_mime_type" not in err_str2:
                    raise err2

        # Tier 3: Try in plain text mode (auto-repaired by clean_and_parse_json)
        if "JSON mode" in err_str1 or "response_mime_type" in err_str1 or "INVALID_ARGUMENT" in err_str1:
            plain_text_contents = []
            for item in fallback_contents:
                if isinstance(item, str):
                    plain_text_contents.append(f"{item}\n\nCRITICAL: Return ONLY a valid JSON object matching the requested schema. Do not include markdown fences or conversational text.")
                else:
                    plain_text_contents.append(item)

            return client.models.generate_content(
                model=model,
                contents=plain_text_contents,
                config={"temperature": temperature}
            )
        raise err1

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
for key in ["pass1_result", "pass2_result", "delivered_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "stats" not in st.session_state:
    st.session_state.stats = UsageStats()

# -----------------------------------------------------------------------------
# Sidebar: Models, Chunking, Settings & Token Calculator
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ QA Configuration")

api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))

discovered_models = fetch_available_gemini_models(api_key) if api_key else []

# Verified production Gemini models
preset_p1_models = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
preset_p2_models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]

all_p1_options = list(dict.fromkeys(preset_p1_models + discovered_models + ["Custom Model Name..."]))
all_p2_options = list(dict.fromkeys(preset_p2_models + discovered_models + ["Custom Model Name..."]))

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Selection")

# Pass 1 Model Selector
p1_model_choice = st.sidebar.selectbox(
    "Pass 1 Model (Audio Ground Truth)",
    all_p1_options,
    index=0,
    help="Gemini 2.5 Flash / 2.0 Flash is recommended for fast, high-accuracy audio transcription."
)
if p1_model_choice == "Custom Model Name...":
    pass1_model = st.sidebar.text_input("Custom Pass 1 Model ID", value="gemini-2.5-flash")
else:
    pass1_model = p1_model_choice

# Pass 2 Model Selector
p2_model_choice = st.sidebar.selectbox(
    "Pass 2 Model (Adjudication Reasoning)",
    all_p2_options,
    index=0,
    help="Gemini 2.5 Pro provides deep reasoning for dialect and linguistic adjudication."
)
if p2_model_choice == "Custom Model Name...":
    pass2_model = st.sidebar.text_input("Custom Pass 2 Model ID", value="gemini-2.5-pro")
else:
    pass2_model = p2_model_choice

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Long Media Chunking (Batching)")
enable_chunking = st.sidebar.toggle(
    "Enable Audio Chunking",
    value=True,
    help="Splits long media into contiguous batches (~7 mins) to avoid max output token limits."
)
chunk_minutes = st.sidebar.slider("Chunk Window (Minutes)", min_value=3, max_value=15, value=7, step=1, disabled=not enable_chunking)

st.sidebar.markdown("---")
st.sidebar.subheader("Dialogue & Variety Settings")
video_ref_id = st.sidebar.text_input("Video Reference ID", value="EP101_AR_TRANSCRIPTION_001")
source_lang = st.sidebar.text_input("Source Language", value="Arabic")
source_variety = st.sidebar.text_input("Source Variety", value="Egyptian Colloquial Arabic (ammiya), not MSA")
frame_rate = st.sidebar.number_input("Frame Rate (fps)", value=24.0, step=0.001, format="%.3f")

# Live Cost Card in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Live Token & Cost Calculator")
cost_info = st.session_state.stats.calculate()
c_col1, c_col2 = st.sidebar.columns(2)
c_col1.metric("Total Tokens", f"{cost_info['total_tokens']:,}")
c_col2.metric("Total Cost", f"${cost_info['total_cost']:.4f}")

with st.sidebar.expander("Cost & Usage Breakdown", expanded=False):
    st.write(f"- **Pass 1 (`{pass1_model}`):** ${cost_info['pass1_cost']:.4f} ({cost_info['pass1_tokens']:,} tokens)")
    st.write(f"- **Pass 2 (`{pass2_model}`):** ${cost_info['pass2_cost']:.4f} ({cost_info['pass2_tokens']:,} tokens)")
    st.write(f"- **Tokens Saved (vs. Raw JSON):** ~{cost_info['tokens_saved_est']:,} tokens")
    if st.button("Reset Token Counter"):
        st.session_state.stats = UsageStats()
        st.rerun()

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
st.title("🎙️ Audio-Truth Transcription QA Studio")
st.caption(f"Pass 1: **{pass1_model}** (Audio Truth) ➔ Pass 2: **{pass2_model}** (Linguistic Adjudicator)")

tab_ingest, tab_pass1, tab_pass2, tab_dashboard = st.tabs([
    "1. Ingest Deliverables",
    "2. Pass 1: Audio Truth (Chunked)",
    "3. Pass 2: Transcription Audit",
    "4. Reviewer Dashboard & Remediation"
])

# =============================================================================
# TAB 1: Ingest
# =============================================================================
with tab_ingest:
    st.subheader("1. Upload Media & Subtitle Deliverable")
    c1, c2 = st.columns(2)
    with c1:
        up_media = st.file_uploader(
            "Upload Source Video / Audio",
            type=["mp4", "mkv", "mov", "wav", "mp3", "m4a"],
            help="Video files are automatically converted to 16kHz mono audio locally before upload to minimize tokens."
        )
    with c2:
        up_sub = st.file_uploader(
            "Upload Transcription Sheet (CSV / Excel)",
            type=["csv", "xlsx"],
            help="Requires columns: Cue ID, Start Time, End Time, Transcription-Human Reviewed, Speaker, Emotion."
        )

    if up_sub:
        try:
            if up_sub.name.endswith(".csv"):
                st.session_state.delivered_df = pd.read_csv(up_sub)
            else:
                st.session_state.delivered_df = pd.read_excel(up_sub)
            st.success(f"✅ Loaded {len(st.session_state.delivered_df)} cue rows from `{up_sub.name}`.")
            with st.expander("Preview Ingested Data", expanded=False):
                st.dataframe(st.session_state.delivered_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

# =============================================================================
# TAB 2: Pass 1 Audio Truth Gloss
# =============================================================================
with tab_pass1:
    st.subheader("2. Blind Audio Gloss & Coverage Sweep")
    st.markdown(f"""
    - **Engine in Use**: `{pass1_model}`
    - **Audio Ground Truth**: Extracts exact verbatim Arabic spoken in audio.
    - **Zero Script Bias**: Model has no access to delivered human transcriptions.
    - **Contiguous Chunking**: Handles long media seamlessly without missing boundaries.
    """)

    if not up_media or st.session_state.delivered_df is None:
        st.warning("⚠️ Please upload both the media file and subtitle sheet in Tab 1.")
    else:
        compact_cues = TokenOptimizer.cues_to_compact_tsv(st.session_state.delivered_df)
        cache_hash = hashlib.sha256(f"{video_ref_id}_{up_media.name}_{compact_cues}_{pass1_model}".encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"pass1_{cache_hash}.json")

        col_p1, col_info = st.columns([1, 2])
        with col_p1:
            run_p1 = st.button("🚀 Run Pass 1 Audio Gloss", type="primary", use_container_width=True)
        with col_info:
            if os.path.exists(cache_file):
                st.info(f"⚡ Cached Pass 1 artifact found (`{cache_hash[:8]}`). Zero additional audio tokens needed.")
                if st.session_state.pass1_result is None:
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            st.session_state.pass1_result = json.load(f)
                    except Exception:
                        pass

        if run_p1:
            if not api_key:
                st.error("❌ Gemini API Key is required. Please set it in the sidebar.")
            else:
                full_audio_path = None
                with st.spinner("Extracting lightweight audio from media..."):
                    try:
                        full_audio_path = extract_compressed_audio(up_media.getvalue(), up_media.name)
                        client = genai.Client(api_key=api_key)

                        if enable_chunking:
                            chunks = create_contiguous_chunks(
                                st.session_state.delivered_df,
                                full_audio_path,
                                chunk_duration_minutes=chunk_minutes,
                                fps=frame_rate
                            )
                            st.write(f"📦 Processing **{len(chunks)} contiguous chunk(s)** (~{chunk_minutes} min each)...")
                        else:
                            chunks = [{
                                "chunk_index": 1,
                                "start_sec": 0.0,
                                "end_sec": None,
                                "duration_sec": None,
                                "start_tc": "Start",
                                "end_tc": "End",
                                "cues_df": st.session_state.delivered_df
                            }]
                            st.write(f"🎧 Auditing entire media in **1 single full pass** ({len(st.session_state.delivered_df)} cues)...")

                        progress_bar = st.progress(0.0)
                        all_cues = []
                        all_uncovered = []

                        for idx, chunk in enumerate(chunks):
                            chunk_cues_df = chunk["cues_df"]
                            
                            if enable_chunking:
                                progress_text = f"Auditing Chunk {idx+1}/{len(chunks)} [{chunk['start_tc']} - {chunk['end_tc']}] ({len(chunk_cues_df)} cues)..."
                                chunk_audio = slice_audio_segment(full_audio_path, chunk["start_sec"], chunk["duration_sec"])
                            else:
                                progress_text = f"Auditing entire media file ({len(chunk_cues_df)} cues)..."
                                chunk_audio = full_audio_path

                            progress_bar.progress(float(idx) / float(len(chunks)), text=progress_text)
                            uploaded_chunk = client.files.upload(file=chunk_audio)
                            
                            chunk_compact_tsv = TokenOptimizer.cues_to_compact_tsv(chunk_cues_df) if not chunk_cues_df.empty else "cue_id|start|end\n"
                            
                            sys_p = prompts.PASS1_SYSTEM.format(
                                source_lang=source_lang,
                                source_variety=source_variety,
                                frame_rate=frame_rate
                            )
                            usr_p = prompts.PASS1_USER.format(
                                video_ref_id=f"{video_ref_id}" if not enable_chunking else f"{video_ref_id}_chunk_{idx+1}",
                                compact_cues=chunk_compact_tsv
                            )

                            res = call_gemini_safe(
                                client=client,
                                model=pass1_model,
                                contents=[uploaded_chunk, usr_p],
                                system_instruction=sys_p,
                                temperature=0.0
                            )

                            if hasattr(res, "usage_metadata"):
                                st.session_state.stats.add_pass1(res.usage_metadata, pass1_model)

                            raw_res_text = safe_extract_text(res)
                            chunk_dict = parse_and_validate_pass1(raw_res_text)
                            all_cues.extend(chunk_dict.get("cues", []))
                            all_uncovered.extend(chunk_dict.get("uncovered_speech", []))

                            if enable_chunking and chunk_audio != full_audio_path and os.path.exists(chunk_audio):
                                try:
                                    os.remove(chunk_audio)
                                except Exception:
                                    pass

                        progress_bar.progress(1.0, text="✅ Audit completed successfully!")

                        stitched_pass1 = {
                            "video_ref_id": video_ref_id,
                            "cues": all_cues,
                            "uncovered_speech": all_uncovered
                        }

                        st.session_state.pass1_result = stitched_pass1
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(stitched_pass1, f, ensure_ascii=False, indent=2)

                        st.success("✅ Pass 1 Audio Ground Truth generated, stitched, and cached!")
                    except Exception as e:
                        st.error(f"Pass 1 execution error: {e}")
                    finally:
                        if full_audio_path and os.path.exists(full_audio_path):
                            try:
                                os.remove(full_audio_path)
                            except Exception:
                                pass

        if st.session_state.pass1_result:
            m1, m2 = st.columns(2)
            cues_count = len(st.session_state.pass1_result.get("cues", []))
            uncovered_count = len(st.session_state.pass1_result.get("uncovered_speech", []))
            m1.metric("Total Cues Glossed", cues_count)
            m2.metric("Uncovered Speech Segments Detected", uncovered_count)

            with st.expander("Inspect Pass 1 Audio Ground Truth JSON", expanded=False):
                st.json(st.session_state.pass1_result)

# =============================================================================
# TAB 3: Pass 2 Transcription Audit
# =============================================================================
with tab_pass2:
    st.subheader("3. Forensic Transcription Audit (Text vs. Audio Truth)")
    st.markdown(f"""
    - **Reasoning Engine**: `{pass2_model}`
    - **Single-Call Holistic Audit**: Evaluates all stitched cues at once to maintain cross-line terminology and speaker consistency.
    - **Token-Optimized**: Uses compact tables and exception-only reporting.
    """)

    if not st.session_state.pass1_result or st.session_state.delivered_df is None:
        st.warning("⚠️ Complete Pass 1 before proceeding to Pass 2.")
    else:
        if st.button("⚖️ Run Pass 2 Transcription Audit", type="primary"):
            if not api_key:
                st.error("❌ Gemini API Key is required.")
            else:
                with st.spinner(f"Adjudicating transcriptions against Audio Truth with {pass2_model}..."):
                    try:
                        compact_p1 = TokenOptimizer.compact_pass1_for_transcription_qa(st.session_state.pass1_result)
                        compact_deliv = TokenOptimizer.delivered_to_compact_transcription_table(st.session_state.delivered_df)

                        sys_p = prompts.PASS2_SYSTEM.format(
                            source_lang=source_lang,
                            source_variety=source_variety
                        )
                        usr_p = prompts.PASS2_USER.format(
                            video_ref_id=video_ref_id,
                            compact_pass1=compact_p1,
                            compact_delivered=compact_deliv
                        )

                        client = genai.Client(api_key=api_key)
                        res = call_gemini_safe(
                            client=client,
                            model=pass2_model,
                            contents=[usr_p],
                            system_instruction=sys_p,
                            temperature=0.0
                        )

                        if hasattr(res, "usage_metadata"):
                            st.session_state.stats.add_pass2(res.usage_metadata, pass2_model)

                        raw_res_text = safe_extract_text(res)
                        validated_dict = parse_and_validate_pass2(raw_res_text)
                        st.session_state.pass2_result = validated_dict
                        st.success("✅ Transcription Audit Complete! View findings in Tab 4.")
                    except Exception as e:
                        st.error(f"Pass 2 execution error: {e}")

        if st.session_state.pass2_result:
            st.json(st.session_state.pass2_result.get("summary", {}))

# =============================================================================
# TAB 4: Reviewer Dashboard & Remediation Sheet
# =============================================================================
with tab_dashboard:
    if not st.session_state.pass2_result:
        st.info("ℹ️ No audit report loaded. Run Pass 2 to generate transcription findings.")
    else:
        report = st.session_state.pass2_result
        summary = report.get("summary", {})

        # Verdict
        verdict = summary.get("verdict", "UNKNOWN")
        v_color = "green" if verdict == "PASS" else "orange" if verdict == "PASS_WITH_FIXES" else "red"
        st.markdown(f"## Overall Transcription Verdict: :{v_color}[{verdict}]")
        if summary.get("verdict_reason"):
            st.write(f"**Verdict Reason:** {summary.get('verdict_reason')}")

        # Metrics Bar
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Audited Cues", summary.get("lines_audited", 0))
        c2.metric("Clean Lines", summary.get("lines_clean", 0))
        c3.metric("🔴 Critical", summary.get("counts_by_severity", {}).get("CRITICAL", 0))
        c4.metric("🟠 Major", summary.get("counts_by_severity", {}).get("MAJOR", 0))
        c5.metric("🟡 Minor", summary.get("counts_by_severity", {}).get("MINOR", 0))

        # Error Breakdown Chart
        st.markdown("### 📊 Error Breakdown by Type")
        err_counts = summary.get("counts_by_error_type", {})
        if any(err_counts.values()):
            st.bar_chart(pd.DataFrame([err_counts]))

        # Systemic Issues
        if report.get("systemic_transcription_issues"):
            st.markdown("### ⚠️ Systemic Transcription Patterns")
            for item in report["systemic_transcription_issues"]:
                with st.chat_message("assistant"):
                    st.markdown(f"**Pattern:** {item.get('pattern')} | **Type:** `{item.get('error_type')}`")
                    cues_list = ", ".join(str(c) for c in item.get("affected_cue_ids", []))
                    if cues_list:
                        st.markdown(f"**Affected Cues:** {cues_list}")
                    st.info(f"💡 **Recommendation:** {item.get('recommendation')}")

        # Missing Dialogue
        if report.get("missing_dialogue"):
            st.markdown("### 🔇 Missing Uncovered Dialogue")
            for md in report["missing_dialogue"]:
                st.error(
                    f"**Window [{md.get('approx_start')} - {md.get('approx_end')}]** (Severity: {md.get('severity')})\n\n"
                    f"**Spoken in Audio:** `{md.get('audio_verbatim')}` ({md.get('audio_gloss')})\n\n"
                    f"**Explanation:** {md.get('explanation')}\n\n"
                    f"**Suggested Cue Transcription:** `{md.get('suggested_transcription')}`"
                )

        # Flagged Lines
        st.markdown("### 🚩 Flagged Transcription Lines")
        flagged = report.get("flagged_lines", [])

        if not flagged:
            st.success("🎉 All delivered transcriptions accurately match the spoken audio truth!")
        else:
            for line in flagged:
                cue_title = f"Cue `{line.get('cue_id')}` [{line.get('start')} - {line.get('end')}] — {line.get('speaker', 'Unknown')}"
                with st.expander(cue_title, expanded=True):
                    col_aud, col_del = st.columns(2)
                    with col_aud:
                        st.markdown("**🎧 What Was Spoken in Audio (Verbatim & Gloss):**")
                        st.code(line.get("audio_verbatim", ""), language="text")
                        st.caption(f"English Meaning: {line.get('audio_gloss', '')}")

                    with col_del:
                        st.markdown("**📄 Delivered Human Transcription:**")
                        st.code(line.get("delivered_transcription", ""), language="text")

                    st.markdown("##### Transcription Findings:")
                    for f in line.get("findings", []):
                        badge = "red" if f.get("severity") == "CRITICAL" else "orange" if f.get("severity") == "MAJOR" else "blue"
                        st.markdown(f"- :{badge}[**{f.get('severity')}**] `{f.get('error_type')}` (Confidence: {f.get('confidence', 1.0)})")
                        st.write(f"  *Explanation:* {f.get('explanation')}")
                        if f.get("audio_evidence") or f.get("transcription_evidence"):
                            st.write(f"  *Evidence:* Audio said `{f.get('audio_evidence')}`, text wrote `{f.get('transcription_evidence')}`")
                        if f.get("suggested_transcription_fix"):
                            st.success(f"  *Suggested Arabic Fix:* `{f.get('suggested_transcription_fix')}`")
                        st.divider()

        # Export Options
        st.markdown("### 📥 Export Transcription Remediation Sheet")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="Download Full Audit (JSON)",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name=f"Transcription_QA_{video_ref_id}.json",
                mime="application/json",
                use_container_width=True
            )
        with dl_col2:
            flattened = []
            for line in flagged:
                for f in line.get("findings", []):
                    flattened.append({
                        "Cue ID": line.get("cue_id"),
                        "Start Time": line.get("start"),
                        "End Time": line.get("end"),
                        "Speaker": line.get("speaker"),
                        "Audio Verbatim": line.get("audio_verbatim"),
                        "Audio English Gloss": line.get("audio_gloss"),
                        "Delivered Transcription": line.get("delivered_transcription"),
                        "Severity": f.get("severity"),
                        "Error Type": f.get("error_type"),
                        "Explanation": f.get("explanation"),
                        "Suggested Correction": f.get("suggested_transcription_fix"),
                    })
            if flattened:
                csv_str = pd.DataFrame(flattened).to_csv(index=False)
                st.download_button(
                    label="Download Transcriber Action Sheet (CSV)",
                    data=csv_str,
                    file_name=f"Transcriber_Action_Sheet_{video_ref_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
