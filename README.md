# 💧 SmartWater Agriculture — Frontend

AI-Powered Groundwater & Irrigation Advisory  
**Igabi & Zaria LGAs · Kaduna State, Nigeria**

Powered by a Bayesian Decision Engine · ABU Zaria & IAR Zaria  
Built for the M4D Open Innovation Challenge 2025/26 — Digital Extension Track

---

## What This Is

SmartWater Agriculture is a farmer-facing advisory tool that uses Bayesian AI to help smallholder farmers in Kaduna State make smarter decisions about:

1. **Irrigation** — when and how much to water crops
2. **Borehole Drilling** — probability of success before investing
3. **Early Warning** — anticipate water stress before it affects crops

---

## Project Structure

```
SmartWater-AI-Frontend/
├── .streamlit/
│   └── config.toml          # Light theme configuration
├── streamlit_app/
│   ├── app.py               # Main application
│   └── requirements.txt     # Python dependencies
└── README.md
```

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Ishmeeeel/SmartWater-AI-Frontend.git
cd SmartWater-AI-Frontend

# Install dependencies
pip install -r streamlit_app/requirements.txt

# Run
streamlit run streamlit_app/app.py
```

---

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select this repo
4. **Main file path:** `streamlit_app/app.py`
5. Deploy

---

## Environment Variables

Set in Streamlit Cloud → App Settings → Secrets:

```toml
API_BASE_URL = "https://smartwater-api.onrender.com"
```

---

## Demo Mode

The app runs fully in **Demo Mode** without the backend connected.  
All three advisory modules show realistic outputs using the built-in  
Bayesian CPT fallback logic from the SmartWater model.

Connect the backend API (`SmartWater-AI-Backend` repo) to switch to live predictions.

---

## Languages

- 🇬🇧 English
- 🇳🇬 Hausa (Hausa)

Toggle via the sidebar language selector.

---

## Backend

See: **SmartWater-AI-Backend** (separate repo)  
API endpoints: `/predict-irrigation`, `/predict-drilling`, `/predict-warning`

---

## Partners

- Ahmadu Bello University, Zaria
- Institute for Agricultural Research (IAR), Zaria
- Achesae Farmers NGO, Kaduna State