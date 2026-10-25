from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

import streamlit as st

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


APP_TITLE = "Multimodal Construction Intelligence Platform"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MODEL_NAME = "gemini-2.5-flash"

STRUCTURE_RATES = {
    "Residential": {"build": 2200, "steel": 4.0, "cement": 0.40, "value": 3000},
    "Commercial": {"build": 3000, "steel": 5.0, "cement": 0.48, "value": 4200},
    "Industrial": {"build": 2600, "steel": 4.5, "cement": 0.44, "value": 3400},
}
LOCATION_MULTIPLIERS = {"Tier 1": 1.25, "Tier 2": 1.0, "Tier 3": 0.82}

SAFETY_SCHEMA = {
    "safety_score": 0,
    "overall_status": "Needs attention",
    "ppe_detection": [
        {"item": "Helmet", "status": "Observed", "notes": "Visible/not visible/unclear"}
    ],
    "hazards": [
        {
            "hazard": "Description",
            "severity": "Low",
            "remedy": "Recommended corrective action",
        }
    ],
    "limitations": ["Visual inspection limitations"],
}
CRACK_SCHEMA = {
    "defect_type": "Hairline",
    "estimated_width_mm": 0.5,
    "root_causes": ["Possible cause"],
    "recommended_repairs": ["Recommended engineering assessment or repair"],
    "urgency": "Monitor",
    "observations": ["Visual observation"],
    "limitations": ["Width estimate is visual and requires on-site measurement"],
}


def _clean_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    result = json.loads(cleaned)
    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")
    return result


def _gemini_client(api_key: str) -> Any:
    if genai is None:
        raise RuntimeError("The google-genai package is not installed.")
    return genai.Client(api_key=api_key)


def _generate_json(api_key: str, prompt: str, image_bytes: bytes, image_mime: str) -> dict[str, Any]:
    client = _gemini_client(api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return _clean_json_response(response.text)


def _safety_report(api_key: str | None, image_bytes: bytes, image_mime: str) -> dict[str, Any]:
    prompt = f"""
Inspect this construction-site image for visible PPE and obvious site hazards. Return
only a JSON object matching this shape: {json.dumps(SAFETY_SCHEMA)}.
Include these PPE items in ppe_detection: helmet, high-visibility vest, steel-toe
boots, and safety harness. Use status values Observed, Not observed, or Unclear.
Do not claim that footwear or harness use is verified when the image cannot establish
it. Score visible conditions from 0 to 100 and use overall_status (Compliant,
Needs attention, or Critical). Each hazard severity must be Low, Medium, High, or
Critical. Give practical corrective remedies. Mention that this is a limited visual
screening, not an OSHA/IS 456 compliance certification; IS 456 is a concrete code and
does not itself define a complete PPE checklist. Do not infer unseen conditions.
"""
    if api_key:
        report = _generate_json(api_key, prompt, image_bytes, image_mime)
        return _validate_safety_report(report)
    return {
        "safety_score": 72,
        "overall_status": "Needs attention",
        "ppe_detection": [
            {"item": "Helmet", "status": "Unclear", "notes": "Demo result; image was not analyzed."},
            {"item": "High-visibility vest", "status": "Unclear", "notes": "Demo result; image was not analyzed."},
            {"item": "Steel-toe boots", "status": "Unclear", "notes": "Cannot be verified from this demo."},
            {"item": "Safety harness", "status": "Unclear", "notes": "Demo result; image was not analyzed."},
        ],
        "hazards": [
            {
                "hazard": "Example: unprotected edge or trip hazard",
                "severity": "Medium",
                "remedy": "Demo recommendation only: inspect the area and install appropriate controls.",
            }
        ],
        "limitations": ["Demonstration data only; the uploaded image was not analyzed."],
    }


def _validate_safety_report(report: dict[str, Any]) -> dict[str, Any]:
    score = report.get("safety_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
        raise ValueError("Gemini returned an invalid safety_score (expected 0-100).")
    if not isinstance(report.get("ppe_detection"), list) or not isinstance(report.get("hazards"), list):
        raise ValueError("Gemini returned an invalid safety report structure.")
    allowed_severities = {"Low", "Medium", "High", "Critical"}
    if any(not isinstance(item, dict) or item.get("severity") not in allowed_severities for item in report["hazards"]):
        raise ValueError("Gemini returned an invalid hazard severity.")
    return report


def _crack_report(api_key: str | None, image_bytes: bytes, image_mime: str) -> dict[str, Any]:
    prompt = f"""
Visually inspect this concrete or wall surface and return only JSON matching:
{json.dumps(CRACK_SCHEMA)}. Classify defect_type as Hairline, Settlement,
Structural Shear, Thermal Expansion, Spalling, or Unclear. Estimate width_mm only
as a rough visual estimate; if there is no scale reference, clearly state that a
reliable measurement is not possible. Include plausible root causes, conservative
recommended engineering repairs (e.g. epoxy injection only if appropriate after
assessment, or structural re-grouting where suitable), and urgency (Monitor,
Schedule inspection, Urgent inspection). Never diagnose structural safety from one
photo. Explicitly recommend an on-site inspection by a qualified structural engineer
for suspected structural damage.
"""
    if api_key:
        report = _generate_json(api_key, prompt, image_bytes, image_mime)
        if report.get("defect_type") not in {
            "Hairline", "Settlement", "Structural Shear", "Thermal Expansion", "Spalling", "Unclear"
        }:
            raise ValueError("Gemini returned an invalid defect_type.")
        width = report.get("estimated_width_mm")
        if width is not None and (
            isinstance(width, bool) or not isinstance(width, (int, float)) or width < 0
        ):
            raise ValueError("Gemini returned an invalid estimated crack width.")
        return report
    return {
        "defect_type": "Hairline",
        "estimated_width_mm": 0.5,
        "root_causes": ["Demo example: drying shrinkage or surface-level movement."],
        "recommended_repairs": [
            "Demo recommendation only: document and monitor; have a qualified engineer assess before repair."
        ],
        "urgency": "Monitor",
        "observations": ["Demo result; the uploaded image was not analyzed."],
        "limitations": ["Illustrative result only; no image analysis or physical measurement was performed."],
    }


def _format_inr(lakhs: float) -> str:
    if lakhs >= 100:
        return f"₹{lakhs / 100:,.2f} Cr"
    return f"₹{lakhs:,.2f} L"


def _generate_concept_render(prompt: str) -> bytes:
    encoded_prompt = urllib.parse.quote(prompt, safe="")
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        "?width=1024&height=768&model=flux&nologo=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "ConstructionIntelligencePlatform/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("image/"):
            raise ValueError(f"Image provider returned unexpected content type: {content_type}.")
        image_bytes = response.read(20 * 1024 * 1024 + 1)
    if not image_bytes or len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("Generated image was empty or exceeded the 20 MB limit.")
    return image_bytes


def _weather_advice(condition: str) -> tuple[str, list[str]]:
    advice = {
        "High Wind": (
            "Pause wind-sensitive work and reassess site conditions.",
            [
                "Suspend crane and lifting operations when wind exceeds the equipment/manufacturer limit.",
                "Secure loose materials, temporary works, and elevated platforms.",
                "Restrict access to exposed edges and reassess before resuming work.",
            ],
        ),
        "Heavy Rain": (
            "Prioritize electrical isolation, drainage, and slip prevention.",
            [
                "Stop exposed electrical work; isolate and inspect affected circuits and tools.",
                "Clear drainage safely and prevent excavation entry where flooding or instability is possible.",
                "Protect fresh concrete from washout and follow the approved mix and curing plan.",
            ],
        ),
        "Extreme Heat": (
            "Reduce heat exposure and plan hydration/rest breaks.",
            [
                "Schedule strenuous tasks during cooler hours and provide shaded rest areas.",
                "Provide drinking water and monitor workers for heat-stress symptoms.",
                "Protect and cure concrete per the approved method statement; avoid uncontrolled rapid drying.",
            ],
        ),
        "Normal": (
            "Continue routine controls and verify conditions at the start of each shift.",
            [
                "Conduct a toolbox talk and confirm PPE before work begins.",
                "Inspect access routes, lifting equipment, temporary works, and housekeeping.",
                "Follow approved concrete placement, finishing, and curing procedures.",
            ],
        ),
    }
    return advice[condition]


def _assistant_reply(question: str, weather: str, api_key: str | None) -> str:
    if api_key:
        client = _gemini_client(api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=(
                "You are a cautious construction site planning assistant. Provide concise, "
                "practical general guidance, state assumptions, and defer structural, "
                "electrical, legal, and emergency decisions to qualified professionals. "
                "Current weather condition: " + weather + ". User question: " + question
            ),
        )
        if not response.text:
            raise ValueError("Gemini returned an empty chat response.")
        return response.text
    normalized = question.lower()
    if any(word in normalized for word in ("crack", "structural", "column", "beam")):
        return (
            "Demo guidance: restrict access if there are signs of movement, falling material, "
            "or sudden change, and arrange an urgent on-site assessment by a qualified "
            "structural engineer. A chat response or photo cannot establish structural safety."
        )
    if any(word in normalized for word in ("cement", "concrete", "curing")):
        return (
            "Demo guidance: follow the approved mix design and project method statement. "
            "Protect placed concrete from weather and maintain the specified curing regime; "
            "confirm requirements with the site engineer."
        )
    return (
        f"Demo guidance for {weather.lower()} conditions: follow the approved site method "
        "statement, conduct a shift-start risk assessment, and consult the responsible "
        "site engineer for task-specific or safety-critical decisions."
    )


def _check_uploaded_image(upload: Any) -> tuple[bytes, str]:
    if upload is None:
        raise ValueError("Choose an image before running the analysis.")
    if upload.type not in {"image/jpeg", "image/png"}:
        raise ValueError("Upload a JPG or PNG image.")
    image_bytes = upload.getvalue()
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be non-empty and no larger than 10 MB.")
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Upload a JPG or PNG image.")
            image_mime = "image/jpeg" if image.format == "JPEG" else "image/png"
            image.verify()
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    return image_bytes, image_mime


def _show_json_report(report: dict[str, Any], name: str) -> None:
    st.json(report)
    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name=f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        mime="application/json",
        key=f"download-{name}",
    )


def _render_safety_tab(api_key: str | None) -> None:
    st.subheader("Site Safety & PPE Audit")
    st.caption("Advisory visual screening only. It is not a safety certification or a substitute for an on-site inspection.")
    upload = st.file_uploader("Upload a construction-site photo", type=["jpg", "jpeg", "png"], key="safety-image")
    if st.button("Analyze site safety", type="primary", key="run-safety"):
        try:
            image_bytes, image_mime = _check_uploaded_image(upload)
            with st.spinner("Reviewing visible PPE and hazards..."):
                report = _safety_report(api_key, image_bytes, image_mime)
            st.session_state["safety_report"] = report
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            st.error(f"Safety analysis could not be completed: {exc}")
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")
    report = st.session_state.get("safety_report")
    if report:
        score_col, status_col = st.columns(2)
        score_col.metric("Safety score", f"{report['safety_score']}/100")
        status_col.metric("Overall status", report.get("overall_status", "Not provided"))
        st.markdown("#### PPE checklist")
        for item in report["ppe_detection"]:
            st.write(f"**{item.get('item', 'PPE item')} — {item.get('status', 'Unclear')}**")
            if item.get("notes"):
                st.caption(item["notes"])
        st.markdown("#### Identified hazards")
        if report["hazards"]:
            for hazard in report["hazards"]:
                st.warning(
                    f"{hazard.get('severity', 'Unrated')}: {hazard.get('hazard', 'Hazard not specified')}. "
                    f"Remedy: {hazard.get('remedy', 'Consult the site safety officer.')}"
                )
        else:
            st.success("No hazards were identified in the visible area. This does not confirm the site is hazard-free.")
        if report.get("limitations"):
            st.caption("Limitations: " + " ".join(report["limitations"]))
        _show_json_report(report, "site-safety-report")


def _render_crack_tab(api_key: str | None) -> None:
    st.subheader("Crack Diagnostics & Structural Integrity")
    st.caption("Visual triage only. A photo cannot establish crack depth, cause, or structural safety.")
    upload = st.file_uploader("Upload a concrete or wall surface photo", type=["jpg", "jpeg", "png"], key="crack-image")
    if st.button("Assess visible defect", type="primary", key="run-crack"):
        try:
            image_bytes, image_mime = _check_uploaded_image(upload)
            with st.spinner("Reviewing the visible surface..."):
                report = _crack_report(api_key, image_bytes, image_mime)
            st.session_state["crack_report"] = report
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            st.error(f"Crack assessment could not be completed: {exc}")
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")
    report = st.session_state.get("crack_report")
    if report:
        first, second = st.columns(2)
        first.metric("Defect classification", report.get("defect_type", "Unclear"))
        width = report.get("estimated_width_mm")
        second.metric("Estimated width", f"{width:g} mm" if isinstance(width, (int, float)) else "Not estimated")
        st.metric("Urgency", report.get("urgency", "Not provided"))
        st.markdown("**Possible root causes**")
        st.write(report.get("root_causes", []))
        st.markdown("**Recommended next steps / repairs**")
        st.write(report.get("recommended_repairs", []))
        if report.get("observations"):
            st.markdown("**Observations**")
            st.write(report["observations"])
        if report.get("limitations"):
            st.caption("Limitations: " + " ".join(report["limitations"]))
        _show_json_report(report, "crack-assessment")


def _render_planning_tab() -> None:
    st.subheader("Planning & Real Estate Valuation")
    st.caption("Indicative planning estimates using transparent built-in assumptions; not a contractor quote or formal valuation.")
    col_a, col_b = st.columns(2)
    structure = col_a.selectbox("Structure type", list(STRUCTURE_RATES))
    location = col_b.selectbox("Location tier", list(LOCATION_MULTIPLIERS))
    area = st.number_input("Built-up area (sq ft)", min_value=100, max_value=2_000_000, value=2000, step=100)
    completion = st.slider("Completion", min_value=0, max_value=100, value=25, format="%d%%")

    rates = STRUCTURE_RATES[structure]
    multiplier = LOCATION_MULTIPLIERS[location]
    budget = area * rates["build"] * multiplier
    incurred = budget * completion / 100
    market_value = area * rates["value"] * multiplier
    steel_kg = area * rates["steel"]
    cement_bags = area * rates["cement"]
    st.progress(completion / 100, text=f"Project milestone: {completion}% complete")
    cards = st.columns(3)
    cards[0].metric("Estimated project budget", _format_inr(budget / 100_000))
    cards[1].metric("Indicative capex incurred", _format_inr(incurred / 100_000))
    cards[2].metric("Indicative asset valuation", _format_inr(market_value / 100_000))
    materials = st.columns(2)
    materials[0].metric("Indicative steel requirement", f"{steel_kg:,.0f} kg")
    materials[1].metric("Indicative cement requirement", f"{cement_bags:,.0f} bags")
    st.caption(
        f"Assumptions: {structure.lower()} construction ₹{rates['build']:,}/sq ft; "
        f"indicative market basis ₹{rates['value']:,}/sq ft; location multiplier "
        f"{multiplier:.2f}; steel {rates['steel']:.1f} kg/sq ft; cement "
        f"{rates['cement']:.2f} bags/sq ft. Validate against local rates, design, and quantity take-offs."
    )

    st.markdown("#### Conceptual 3D rendering prompt")
    rendering_prompt = (
        f"Photorealistic conceptual 3D architectural rendering of a {structure.lower()} "
        f"building, approximately {area:,} sq ft, in an Indian {location.lower()}-tier "
        "city; contemporary climate-responsive design, realistic construction materials, "
        "daylight, street-level perspective. Concept only, not an approved design."
    )
    st.write(rendering_prompt)
    if st.button("Generate conceptual image with Pollinations", key="generate-concept-render"):
        try:
            with st.spinner("Generating a conceptual image with Pollinations..."):
                st.session_state["concept_render"] = _generate_concept_render(rendering_prompt)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            st.error(f"Could not generate the conceptual image: {exc}")
    if st.session_state.get("concept_render"):
        st.image(st.session_state["concept_render"], caption="AI-generated concept only; not an approved construction design.")
    st.caption("Pollinations receives the generated prompt. Do not include private project or client information.")


def _render_weather_tab(api_key: str | None) -> None:
    st.subheader("Daily Site Assistant & Weather-Aware Advisory")
    condition = st.selectbox("Today's site weather", ["Normal", "High Wind", "Heavy Rain", "Extreme Heat"])
    headline, actions = _weather_advice(condition)
    st.info(headline)
    for action in actions:
        st.write(f"- {action}")
    st.caption("Weather guidance is general. Follow local forecasts, site emergency plans, and equipment limits.")

    st.markdown("#### Construction assistant")
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask a construction planning or site-safety question")
    if question:
        st.session_state["chat_messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.spinner("Preparing guidance..."):
                answer = _assistant_reply(question, condition, api_key)
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as exc:
            st.error(f"Assistant request failed: {exc}")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🏗️", layout="wide")
    st.markdown(
        """
        <style>
        :root {
          --ink: #172b3a;
          --muted: #627487;
          --amber: #e9a23b;
          --line: #e4eaf0;
        }
        .stApp {
          background:
            radial-gradient(ellipse at 8% 0%, #e8f0f5 0, transparent 34%),
            #f5f7f9;
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: rgba(245, 247, 249, .86); }
        [data-testid="stMainBlockContainer"] { padding-top: 2.4rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #142b3a 0%, #1d394a 100%);
        }
        [data-testid="stSidebar"] * { color: #eef4f7; }
        [data-testid="stSidebar"] input { color: #172b3a; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #bdcbd4; }
        .hero {
          position: relative;
          overflow: hidden;
          margin: 0 0 1.6rem;
          padding: clamp(1.5rem, 4vw, 2.7rem);
          border: 1px solid #294657;
          border-radius: 22px;
          color: #f7fafc;
          background:
            radial-gradient(circle at 88% 18%, #e9a23b44 0, transparent 24%),
            linear-gradient(115deg, #142b3a 0%, #244657 62%, #31566a 100%);
          box-shadow: 0 18px 42px #172b3a20;
        }
        .hero:after {
          content: "";
          position: absolute;
          right: -45px;
          bottom: -110px;
          width: 290px;
          height: 290px;
          border: 1px solid #ffffff1c;
          border-radius: 50%;
          box-shadow: 0 0 0 26px #ffffff08, 0 0 0 54px #ffffff06;
          pointer-events: none;
        }
        .hero-kicker {
          margin-bottom: .8rem;
          color: #f1bd69;
          font-size: .72rem;
          font-weight: 750;
          letter-spacing: .17em;
          text-transform: uppercase;
        }
        .hero-title {
          max-width: 850px;
          margin: 0;
          color: #fff;
          font-size: clamp(1.8rem, 4vw, 3rem);
          font-weight: 750;
          line-height: 1.12;
          letter-spacing: -.04em;
        }
        .hero-copy {
          max-width: 680px;
          margin: .9rem 0 0;
          color: #d5e0e6;
          font-size: 1rem;
          line-height: 1.65;
        }
        [data-testid="stTabs"] [role="tablist"] {
          gap: .45rem;
          padding: .4rem;
          border: 1px solid var(--line);
          border-radius: 14px;
          background: #fff;
          box-shadow: 0 4px 14px #172b3a08;
        }
        [data-testid="stTabs"] [role="tab"] {
          min-height: 2.8rem;
          padding: 0 1rem;
          border-radius: 10px;
          color: var(--muted);
          font-weight: 650;
          transition: background .18s ease, color .18s ease;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
          color: #fff;
          background: #203c4c;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: #fff; }
        [data-testid="stTabs"] .react-aria-SelectionIndicator { background: var(--amber) !important; }
        [data-testid="stTabPanel"] { padding: 1.5rem .15rem; }
        [data-testid="stTabPanel"] h3 {
          margin-bottom: .35rem;
          color: var(--ink);
          font-size: 1.55rem;
          letter-spacing: -.025em;
        }
        [data-testid="stMetric"] {
          min-height: 112px;
          padding: 1.1rem 1.25rem;
          border: 1px solid var(--line);
          border-radius: 16px;
          background: #fff;
          box-shadow: 0 6px 18px #172b3a08;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
        [data-testid="stMetricValue"] { color: var(--ink); font-weight: 750; }
        [data-testid="stFileUploader"] section {
          border: 1.5px dashed #b8c8d3;
          border-radius: 16px;
          background: #fff;
        }
        [data-testid="stFileUploader"] section:hover { border-color: var(--amber); }
        .stButton > button[kind="primary"] {
          min-height: 2.8rem;
          padding: 0 1.25rem;
          border: 0;
          border-radius: 10px;
          color: #fff;
          background: linear-gradient(105deg, #d88928, #efad4d);
          font-weight: 700;
          box-shadow: 0 5px 14px #d8892830;
          transition: transform .16s ease, box-shadow .16s ease;
        }
        .stButton > button[kind="primary"]:hover {
          transform: translateY(-1px);
          box-shadow: 0 8px 18px #d8892845;
        }
        .stButton > button:not([kind="primary"]) {
          border-radius: 10px;
          border-color: #d7e0e7;
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        [data-testid="stChatMessage"] {
          border: 1px solid var(--line);
          border-radius: 14px;
          background: #fff;
        }
        [data-testid="stProgress"] > div > div > div > div {
          background: linear-gradient(90deg, #d88928, #f0b85e);
        }
        @media (max-width: 700px) {
          [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 3rem; }
          .hero { border-radius: 16px; }
          [data-testid="stTabs"] [role="tab"] { padding: 0 .65rem; font-size: .82rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">Field intelligence · Project clarity</div>
          <h1 class="hero-title">Build safer. Plan smarter.</h1>
          <p class="hero-copy">Multimodal decision support for construction-site safety,
          surface diagnostics, project economics, and weather-aware daily operations.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Google Gemini API key",
            type="password",
            help="Used only for requests initiated in this session. Leave blank for clearly labeled demo mode.",
        ).strip()
        if api_key and genai is None:
            st.error("Install dependencies from requirements.txt to enable Gemini analysis.")
            api_key = ""
        st.caption("API key is kept in session memory and is not written to a database or file.")
        st.divider()
        st.markdown("**Architecture**")
        st.caption("Streamlit UI → Gemini multimodal/text API (optional) → validated JSON reports")
        st.caption("Planning calculations are local; no external database is used.")
        st.markdown("**Features**")
        st.caption("PPE & hazards · Crack triage · Budget & valuation · Weather & chat")
        st.divider()
        st.caption("AI outputs are advisory and may be incorrect. Verify safety-critical decisions with qualified professionals.")

    tabs = st.tabs(
        [
            "🦺  Site Safety",
            "🧱  Crack Diagnostics",
            "📐  Planning & Valuation",
            "🌦️  Daily Assistant",
        ]
    )
    with tabs[0]:
        _render_safety_tab(api_key or None)
    with tabs[1]:
        _render_crack_tab(api_key or None)
    with tabs[2]:
        _render_planning_tab()
    with tabs[3]:
        _render_weather_tab(api_key or None)


if __name__ == "__main__":
    main()
