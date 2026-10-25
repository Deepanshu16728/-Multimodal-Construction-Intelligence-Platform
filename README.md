# Multimodal Construction Intelligence Platform

A Streamlit decision-support app for construction-site image triage, indicative project planning, and daily weather-aware protocols. It runs without a database and supports Gemini analysis when an API key is supplied.

## Run locally

Requires Python 3.9 or newer.

```powershell
cd Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Enter a Google Gemini API key in the sidebar to enable image analysis and the AI assistant. The key is held in Streamlit session memory and is not stored by the app. Without a key, image reports and assistant responses use clearly labeled illustrative demo output; uploaded images are not analyzed in demo mode.

## Streamlit Community Cloud

Deploy this directory, set `app.py` as the app entry point, and add `requirements.txt` as the dependency file. Users can enter their own key in the sidebar. For managed deployments, a `GEMINI_API_KEY` Streamlit secret can also be used by adapting the key input to your deployment's secret-management policy.

## Features and limitations

- **Site safety:** Reviews visible helmets, high-visibility vests, boots, harnesses, and hazards. Results are visual screening only, not an OSHA or IS 456 certification. IS 456 is a concrete code, not a complete PPE checklist.
- **Crack diagnostics:** Provides visual classification, approximate width (if supported), likely causes, and conservative next steps. A photo without a scale cannot establish crack width or structural safety.
- **Planning and valuation:** Shows transparent built-in rate assumptions, indicative capex, material quantities, and a conceptual rendering prompt. Estimates are not contractor quotes, quantity surveys, or formal valuations.
- **Weather and assistant:** Displays general protocols for four weather conditions and provides multi-turn guidance. Follow local forecasts, site procedures, equipment limits, and qualified professional advice.

Image uploads accept JPG/PNG up to 10 MB. Keep API keys private; the app does not persist reports or chat to an external database.
