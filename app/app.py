"""
ClimateAI India - Ultra-Modern Meteorological Intelligence & Telemetry Platform
Equipped with Real-time Sensor Telemetry, 10-Day Multi-Horizon AI Forecasting,
Aerospace & Aviation Turbulence Intelligence, and Realistic Doppler Radar Mapping.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.live_weather import (
    predict_live_weather,
    fetch_live_weather_api,
    fetch_all_50_cities_live,
    fetch_live_air_quality,
    resolve_city_coords,
    compute_aerospace_risk
)

# Page config
st.set_page_config(
    page_title="ClimateAI India | Climate & Aerospace Intelligence Platform",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f30d.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SVG Icons Helper for ultra-clean modern UI
def get_svg_icon(name: str, color: str = "#A78BFA", size: int = 18) -> str:
    icons = {
        "thermometer": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>',
        "droplet": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>',
        "wind": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg>',
        "activity": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "shield-alert": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
        "check-circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "globe": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
        "cpu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "sliders": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
        "bar-chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        "plane": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3.5c-.5-.5-2.5 0-4 1.5L13.5 8.5 5.3 6.7c-.8-.2-1.6.3-1.8 1.1-.2.8.3 1.6 1.1 1.8l6.9 2.5-3.5 3.5-2.6-.6c-.5-.1-1 .1-1.3.5-.4.4-.4 1 0 1.4l2.1 2.1 2.1 2.1c.4.4 1 .4 1.4 0 .4-.3.6-.8.5-1.3l-.6-2.6 3.5-3.5 2.5 6.9c.2.8 1 1.3 1.8 1.1.8-.2 1.3-1 1.1-1.8z"/></svg>',
        "calendar": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
        "radar": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/><path d="M12 6a6 6 0 1 0 6 6 6 6 0 0 0-6-6zm0 10a4 4 0 1 1 4-4 4 4 0 0 1-4 4z"/><line x1="12" y1="12" x2="18.5" y2="5.5"/></svg>',
        "refresh": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>'
    }
    return icons.get(name, icons["activity"])

# Ultra-Modern Obsidian Glass UI Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg-main: #07090F;
        --card-bg: rgba(12, 16, 32, 0.75);
        --card-border: rgba(167, 139, 250, 0.12);
        --accent-primary: #A78BFA;
        --accent-violet: #7C3AED;
        --accent-purple: #C084FC;
        --accent-emerald: #34D399;
        --accent-rose: #F87171;
        --accent-amber: #FBBF24;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main App Background — Live Aurora Mesh */
    .stApp {
        background-color: #070910;
        background-image:
            radial-gradient(ellipse at 15% 20%, rgba(124, 58, 237, 0.22) 0%, transparent 45%),
            radial-gradient(ellipse at 85% 15%, rgba(192, 132, 252, 0.17) 0%, transparent 42%),
            radial-gradient(ellipse at 50% 90%, rgba(52, 211, 153, 0.14) 0%, transparent 48%);
        background-attachment: fixed;
        color: var(--text-primary);
    }

    /* Live animated aurora mesh overlay */
    #live-bg-canvas {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        z-index: 0;
        pointer-events: none;
    }

    /* Typography Hierarchy */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--text-primary);
    }

    .main-header {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
        line-height: 1.2;
    }

    .subtitle {
        font-size: 0.98rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    .section-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 22px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.01em;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }

    .glass-card:hover {
        border-color: rgba(167, 139, 250, 0.35);
        box-shadow: 0 16px 36px -8px rgba(167, 139, 250, 0.15);
        transform: translateY(-2px);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(12, 16, 32, 0.9) 0%, rgba(22, 28, 56, 0.65) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(167, 139, 250, 0.1);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 12px;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg,
            transparent 0%, rgba(167,139,250,0.6) 40%,
            rgba(52,211,153,0.5) 60%, transparent 100%);
        background-size: 200% auto;
        animation: shimmerSweep 3.5s linear infinite;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(167, 139, 250, 0.4);
        box-shadow: 0 12px 30px rgba(167, 139, 250, 0.12);
    }

    .metric-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--text-secondary);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        line-height: 1.15;
    }

    .metric-subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Badges & Chips */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(167, 139, 250, 0.1);
        border: 1px solid rgba(167, 139, 250, 0.25);
        color: var(--accent-primary);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #34D399;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
        animation: pulseAnimation 2s infinite, ringPulse 2s infinite;
        display: inline-block;
    }

    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    @keyframes ringPulse {
        0%   { box-shadow: 0 0 0 0 rgba(52,211,153,0.7); }
        70%  { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(12, 16, 32, 0.92) 0%, rgba(22, 28, 56, 0.75) 50%, rgba(12, 16, 32, 0.95) 100%);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(167, 139, 250, 0.18);
        border-radius: 20px;
        padding: 28px 24px;
        margin-bottom: 20px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
        animation: borderGlow 6s ease-in-out infinite;
    }

    @keyframes borderGlow {
        0%   { border-color: rgba(124, 58, 237, 0.3); box-shadow: 0 0 18px rgba(124, 58, 237, 0.12); }
        33%  { border-color: rgba(192, 132, 252, 0.4); box-shadow: 0 0 24px rgba(192, 132, 252, 0.16); }
        66%  { border-color: rgba(52, 211, 153, 0.3);  box-shadow: 0 0 18px rgba(52, 211, 153, 0.10); }
        100% { border-color: rgba(124, 58, 237, 0.3); box-shadow: 0 0 18px rgba(124, 58, 237, 0.12); }
    }

    /* Hide sidebar completely and top header */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], div[data-testid="collapsedControl"] {
        display: none !important;
    }
    header[data-testid="stHeader"], .stApp > header {
        display: none !important;
    }
    .main .block-container {
        max-width: 96% !important;
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Top Navigation Bar Styling */
    .top-header-bar {
        background: linear-gradient(135deg, rgba(14, 18, 38, 0.92) 0%, rgba(22, 28, 56, 0.8) 100%);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 18px;
        padding: 14px 22px;
        margin-bottom: 14px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    /* Clean horizontal radio navigation */
    div[data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        gap: 8px;
        background: rgba(12, 16, 32, 0.85);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 14px;
        padding: 8px;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    }

    div[data-testid="stRadio"] [role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 8px 18px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: pointer;
        margin: 0;
    }

    div[data-testid="stRadio"] [role="radiogroup"] > label:hover {
        background: rgba(124, 58, 237, 0.18);
        border-color: rgba(167, 139, 250, 0.4);
        color: #FFFFFF;
        transform: translateY(-1px);
    }

    div[data-testid="stRadio"] [role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.35) 0%, rgba(192, 132, 252, 0.4) 100%) !important;
        border-color: rgba(167, 139, 250, 0.65) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 18px rgba(124, 58, 237, 0.35);
    }

    /* Alert Cards */
    .alert-card {
        padding: 14px 18px;
        border-radius: 14px;
        margin: 12px 0;
        display: flex;
        align-items: center;
        gap: 14px;
        backdrop-filter: blur(12px);
    }

    .alert-heatwave {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #FECACA;
    }

    .alert-rain {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.25);
        color: #A7F3D0;
    }

    .alert-normal {
        background: rgba(52, 211, 153, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.2);
        color: #A7F3D0;
    }

    /* 10-Day Forecast Card Grid */
    .day-card {
        background: rgba(14, 19, 38, 0.7);
        border: 1px solid rgba(167, 139, 250, 0.12);
        border-radius: 14px;
        padding: 14px 10px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .day-card:hover {
        border-color: rgba(167, 139, 250, 0.4);
        background: rgba(22, 28, 56, 0.85);
        transform: translateY(-2px);
    }

    /* Flight Corridor Card */
    .flight-route-card {
        background: linear-gradient(135deg, rgba(14, 19, 38, 0.9) 0%, rgba(26, 32, 64, 0.7) 100%);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 12px;
    }

    /* Scrolling live data ticker */
    .ticker-wrapper {
        overflow: hidden;
        background: rgba(10, 13, 28, 0.85);
        border: 1px solid rgba(167, 139, 250, 0.18);
        border-radius: 12px;
        padding: 10px 0;
        margin: 12px 0 16px 0;
        position: relative;
    }
    .ticker-wrapper::before,
    .ticker-wrapper::after {
        content: '';
        position: absolute;
        top: 0; bottom: 0;
        width: 80px;
        z-index: 2;
        pointer-events: none;
    }
    .ticker-wrapper::before {
        left: 0;
        background: linear-gradient(to right, rgba(10,13,28,1), transparent);
    }
    .ticker-wrapper::after {
        right: 0;
        background: linear-gradient(to left, rgba(10,13,28,1), transparent);
    }
    .ticker-track {
        display: flex;
        white-space: nowrap;
        animation: tickerScroll 65s linear infinite;
        gap: 0;
    }
    .ticker-track:hover { animation-play-state: paused; }
    @keyframes tickerScroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0 26px;
        font-size: 0.8rem;
        font-weight: 600;
        border-right: 1px solid rgba(167, 139, 250, 0.12);
        letter-spacing: 0.02em;
    }
    .ticker-label {
        color: #64748B;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Animated logo gradient */
    @keyframes logoShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .logo-gradient {
        background: linear-gradient(135deg, #A78BFA, #C084FC, #34D399, #A78BFA);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: logoShift 5s ease infinite;
    }

    /* ===== ULTRA-SMOOTH ANIMATION SYSTEM ===== */

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.96) translateY(10px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }

    @keyframes radarSpin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes floatSoft {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }

    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 15px rgba(167, 139, 250, 0.15);
        }
        50% {
            box-shadow: 0 0 28px rgba(167, 139, 250, 0.35);
        }
    }

    /* Staggered entrance animations on cards */
    .hero-banner {
        animation: fadeInScale 0.6s cubic-bezier(0.16, 1, 0.3, 1) both, borderGlow 6s ease-in-out infinite;
    }

    .top-header-bar {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    .metric-card {
        animation: fadeInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.28s ease, border-color 0.28s ease;
    }
    .metric-card:nth-of-type(1) { animation-delay: 0.05s; }
    .metric-card:nth-of-type(2) { animation-delay: 0.10s; }
    .metric-card:nth-of-type(3) { animation-delay: 0.15s; }
    .metric-card:nth-of-type(4) { animation-delay: 0.20s; }

    .metric-card:hover {
        transform: translateY(-5px) scale(1.01);
        border-color: rgba(167, 139, 250, 0.45);
        box-shadow: 0 16px 36px -8px rgba(124, 58, 237, 0.25), 0 0 20px rgba(167, 139, 250, 0.15);
    }

    .glass-card {
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.28s ease, border-color 0.28s ease;
    }

    .glass-card:hover {
        transform: translateY(-4px) scale(1.008);
        border-color: rgba(167, 139, 250, 0.4);
        box-shadow: 0 18px 40px -10px rgba(124, 58, 237, 0.22);
    }

    .day-card {
        animation: fadeInScale 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), background 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .day-card:hover {
        transform: translateY(-6px) scale(1.04);
        border-color: rgba(167, 139, 250, 0.5);
        background: linear-gradient(145deg, rgba(22, 28, 56, 0.95) 0%, rgba(32, 40, 78, 0.8) 100%);
        box-shadow: 0 12px 28px rgba(124, 58, 237, 0.28);
    }

    .alert-card {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .alert-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }

    .flight-route-card {
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.28s ease, border-color 0.28s ease, box-shadow 0.28s ease;
    }
    .flight-route-card:hover {
        transform: translateY(-3px);
        border-color: rgba(167, 139, 250, 0.45);
        box-shadow: 0 14px 32px rgba(124, 58, 237, 0.2);
    }

    /* Button Animations */
    .stButton > button {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.25) 0%, rgba(167, 139, 250, 0.3) 100%) !important;
        border: 1px solid rgba(167, 139, 250, 0.35) !important;
        color: #F8FAFC !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.45) 0%, rgba(192, 132, 252, 0.5) 100%) !important;
        border-color: rgba(167, 139, 250, 0.7) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Selectbox animation */
    div[data-baseweb="select"] > div {
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(167, 139, 250, 0.45) !important;
        box-shadow: 0 0 14px rgba(167, 139, 250, 0.15) !important;
    }

    /* Floating ambient orbs */
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(90px);
        pointer-events: none;
        z-index: 0;
        animation: orbDrift var(--dur, 22s) var(--delay, 0s) infinite ease-in-out alternate;
    }
    @keyframes orbDrift {
        0%   { transform: translate(0px, 0px) scale(1);    opacity: var(--op-lo, 0.07); }
        33%  { transform: translate(var(--x1,  60px), var(--y1, -80px)) scale(1.12); opacity: var(--op-hi, 0.13); }
        66%  { transform: translate(var(--x2, -50px), var(--y2,  50px)) scale(0.92); opacity: var(--op-lo, 0.07); }
        100% { transform: translate(var(--x3,  30px), var(--y3, -30px)) scale(1.05); opacity: var(--op-hi, 0.11); }
    }

    /* Rising atmospheric particle dots */
    .particle {
        position: fixed;
        width: 2.5px;
        height: 2.5px;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: particleRise var(--pdur, 18s) var(--pdelay, 0s) infinite linear;
        opacity: 0;
    }
    @keyframes particleRise {
        0%   { transform: translateY(100vh) translateX(0);   opacity: 0; }
        10%  { opacity: var(--pop, 0.5); }
        90%  { opacity: var(--pop, 0.4); }
        100% { transform: translateY(-10vh) translateX(var(--drift, 30px)); opacity: 0; }
    }

    @keyframes shimmerSweep {
        0%   { background-position: -200% center; }
        100% { background-position:  200% center; }
    }
</style>
""", unsafe_allow_html=True)

# === LIVE ANIMATED BACKGROUND — Canvas + CSS Particle System ===
st.markdown("""
<canvas id="live-bg-canvas"></canvas>
<style>
  /* Animated aurora mesh blobs */
  .aurora-blob {
      position: fixed;
      border-radius: 50%;
      filter: blur(80px);
      pointer-events: none;
      z-index: 0;
      mix-blend-mode: screen;
  }
  @keyframes aurora1 {
      0%   { transform: translate(0,0) scale(1); opacity: 0.22; }
      25%  { transform: translate(120px,-80px) scale(1.2); opacity: 0.30; }
      50%  { transform: translate(-60px,100px) scale(0.9); opacity: 0.18; }
      75%  { transform: translate(80px,60px) scale(1.15); opacity: 0.28; }
      100% { transform: translate(0,0) scale(1); opacity: 0.22; }
  }
  @keyframes aurora2 {
      0%   { transform: translate(0,0) scale(1); opacity: 0.18; }
      33%  { transform: translate(-100px,70px) scale(1.18); opacity: 0.26; }
      66%  { transform: translate(70px,-90px) scale(0.85); opacity: 0.15; }
      100% { transform: translate(0,0) scale(1); opacity: 0.18; }
  }
  @keyframes aurora3 {
      0%   { transform: translate(0,0) scale(1); opacity: 0.14; }
      40%  { transform: translate(90px,80px) scale(1.22); opacity: 0.22; }
      80%  { transform: translate(-80px,-60px) scale(0.88); opacity: 0.12; }
      100% { transform: translate(0,0) scale(1); opacity: 0.14; }
  }
  @keyframes aurora4 {
      0%   { transform: translate(0,0) scale(1); opacity: 0.16; }
      50%  { transform: translate(-130px,50px) scale(1.3); opacity: 0.24; }
      100% { transform: translate(0,0) scale(1); opacity: 0.16; }
  }
  /* Atmospheric grid lines */
  .atmo-grid {
      position: fixed;
      top: 0; left: 0;
      width: 100vw; height: 100vh;
      pointer-events: none;
      z-index: 0;
      background-image:
          linear-gradient(rgba(167,139,250,0.035) 1px, transparent 1px),
          linear-gradient(90deg, rgba(167,139,250,0.035) 1px, transparent 1px);
      background-size: 60px 60px;
      animation: gridShift 20s linear infinite;
  }
  @keyframes gridShift {
      0%   { background-position: 0 0; }
      100% { background-position: 60px 60px; }
  }
  /* Radar sweep ring */
  .radar-ring {
      position: fixed;
      border-radius: 50%;
      border: 1px solid rgba(52,211,153,0.12);
      pointer-events: none;
      z-index: 0;
      top: 50%; left: 50%;
      transform: translate(-50%,-50%) scale(0);
      animation: radarExpand var(--rdur,8s) var(--rdelay,0s) linear infinite;
  }
  @keyframes radarExpand {
      0%   { transform: translate(-50%,-50%) scale(0);   opacity: 0.35; }
      70%  { opacity: 0.12; }
      100% { transform: translate(-50%,-50%) scale(3.5); opacity: 0; }
  }
  /* Bright floating weather particles */
  .wx-particle {
      position: fixed;
      border-radius: 50%;
      pointer-events: none;
      z-index: 0;
      animation: wxRise var(--wdur,16s) var(--wdelay,0s) infinite linear;
      opacity: 0;
  }
  @keyframes wxRise {
      0%   { transform: translateY(105vh) translateX(0px); opacity: 0; }
      8%   { opacity: var(--wop,0.75); }
      92%  { opacity: var(--wop,0.65); }
      100% { transform: translateY(-8vh) translateX(var(--wdrift,20px)); opacity: 0; }
  }
  /* Shooting star / data stream lines */
  .data-stream {
      position: fixed;
      width: 1.5px;
      border-radius: 2px;
      pointer-events: none;
      z-index: 0;
      opacity: 0;
      animation: streamFall var(--sdur,6s) var(--sdelay,0s) infinite linear;
  }
  @keyframes streamFall {
      0%   { transform: translateY(-60px) scaleY(0); opacity: 0; }
      10%  { opacity: var(--sop,0.6); transform: translateY(0) scaleY(1); }
      80%  { opacity: var(--sop,0.5); }
      100% { transform: translateY(110vh) scaleY(1); opacity: 0; }
  }
</style>

<!-- Aurora blobs -->
<div class="aurora-blob" style="width:750px;height:750px;background:radial-gradient(circle,#7C3AED,transparent 70%);top:-200px;left:-250px;animation:aurora1 26s ease-in-out infinite;"></div>
<div class="aurora-blob" style="width:620px;height:620px;background:radial-gradient(circle,#34D399,transparent 70%);bottom:-180px;right:-200px;animation:aurora2 32s ease-in-out infinite;"></div>
<div class="aurora-blob" style="width:480px;height:480px;background:radial-gradient(circle,#C084FC,transparent 70%);top:40%;left:50%;animation:aurora3 22s ease-in-out infinite;"></div>
<div class="aurora-blob" style="width:400px;height:400px;background:radial-gradient(circle,#0EA5E9,transparent 70%);top:20%;right:5%;animation:aurora4 28s ease-in-out infinite;"></div>
<div class="aurora-blob" style="width:350px;height:350px;background:radial-gradient(circle,#FBBF24,transparent 70%);bottom:30%;left:15%;animation:aurora2 19s ease-in-out infinite reverse;"></div>

<!-- Atmospheric scrolling grid -->
<div class="atmo-grid"></div>

<!-- Radar sweep rings (centered, multiple phases) -->
<div class="radar-ring" style="width:400px;height:400px;--rdur:9s;--rdelay:0s;"></div>
<div class="radar-ring" style="width:400px;height:400px;--rdur:9s;--rdelay:-3s;"></div>
<div class="radar-ring" style="width:400px;height:400px;--rdur:9s;--rdelay:-6s;"></div>
<div class="radar-ring" style="width:600px;height:600px;--rdur:12s;--rdelay:-2s;border-color:rgba(167,139,250,0.10);"></div>
<div class="radar-ring" style="width:600px;height:600px;--rdur:12s;--rdelay:-6s;border-color:rgba(167,139,250,0.10);"></div>
<div class="radar-ring" style="width:600px;height:600px;--rdur:12s;--rdelay:-10s;border-color:rgba(167,139,250,0.10);"></div>

<!-- Dense weather particles -->
<div class="wx-particle" style="left:4%;  width:3px;height:3px;background:#A78BFA;box-shadow:0 0 6px 2px #A78BFA;--wdur:18s;--wdelay:0s;   --wop:0.80;--wdrift:-20px;"></div>
<div class="wx-particle" style="left:10%; width:2px;height:2px;background:#34D399;box-shadow:0 0 5px 2px #34D399;--wdur:14s;--wdelay:-2s;  --wop:0.70;--wdrift: 25px;"></div>
<div class="wx-particle" style="left:17%; width:3px;height:3px;background:#C084FC;box-shadow:0 0 7px 2px #C084FC;--wdur:22s;--wdelay:-5s;  --wop:0.75;--wdrift:-30px;"></div>
<div class="wx-particle" style="left:24%; width:2px;height:2px;background:#FBBF24;box-shadow:0 0 5px 2px #FBBF24;--wdur:17s;--wdelay:-8s;  --wop:0.65;--wdrift: 18px;"></div>
<div class="wx-particle" style="left:31%; width:3px;height:3px;background:#A78BFA;box-shadow:0 0 6px 2px #A78BFA;--wdur:20s;--wdelay:-1s;  --wop:0.80;--wdrift:-22px;"></div>
<div class="wx-particle" style="left:38%; width:2px;height:2px;background:#34D399;box-shadow:0 0 5px 2px #34D399;--wdur:16s;--wdelay:-6s;  --wop:0.70;--wdrift: 35px;"></div>
<div class="wx-particle" style="left:45%; width:3px;height:3px;background:#F87171;box-shadow:0 0 7px 2px #F87171;--wdur:23s;--wdelay:-10s; --wop:0.75;--wdrift:-15px;"></div>
<div class="wx-particle" style="left:52%; width:2px;height:2px;background:#C084FC;box-shadow:0 0 5px 2px #C084FC;--wdur:15s;--wdelay:-3s;  --wop:0.70;--wdrift: 28px;"></div>
<div class="wx-particle" style="left:59%; width:3px;height:3px;background:#FBBF24;box-shadow:0 0 6px 2px #FBBF24;--wdur:19s;--wdelay:-7s;  --wop:0.80;--wdrift:-25px;"></div>
<div class="wx-particle" style="left:66%; width:2px;height:2px;background:#A78BFA;box-shadow:0 0 5px 2px #A78BFA;--wdur:21s;--wdelay:-4s;  --wop:0.65;--wdrift: 20px;"></div>
<div class="wx-particle" style="left:73%; width:3px;height:3px;background:#34D399;box-shadow:0 0 7px 2px #34D399;--wdur:13s;--wdelay:-9s;  --wop:0.75;--wdrift:-28px;"></div>
<div class="wx-particle" style="left:80%; width:2px;height:2px;background:#C084FC;box-shadow:0 0 5px 2px #C084FC;--wdur:24s;--wdelay:-2s;  --wop:0.70;--wdrift: 32px;"></div>
<div class="wx-particle" style="left:87%; width:3px;height:3px;background:#F87171;box-shadow:0 0 6px 2px #F87171;--wdur:17s;--wdelay:-5s;  --wop:0.80;--wdrift:-18px;"></div>
<div class="wx-particle" style="left:94%; width:2px;height:2px;background:#FBBF24;box-shadow:0 0 5px 2px #FBBF24;--wdur:20s;--wdelay:-11s; --wop:0.65;--wdrift: 24px;"></div>

<!-- Vertical data-stream lines -->
<div class="data-stream" style="left:12%;height:120px;background:linear-gradient(to bottom,transparent,#A78BFA,transparent);--sdur:7s;--sdelay:0s;  --sop:0.55;"></div>
<div class="data-stream" style="left:28%;height:90px; background:linear-gradient(to bottom,transparent,#34D399,transparent);--sdur:9s;--sdelay:-2s; --sop:0.50;"></div>
<div class="data-stream" style="left:45%;height:140px;background:linear-gradient(to bottom,transparent,#C084FC,transparent);--sdur:6s;--sdelay:-4s; --sop:0.55;"></div>
<div class="data-stream" style="left:62%;height:100px;background:linear-gradient(to bottom,transparent,#FBBF24,transparent);--sdur:8s;--sdelay:-1s; --sop:0.45;"></div>
<div class="data-stream" style="left:78%;height:110px;background:linear-gradient(to bottom,transparent,#F87171,transparent);--sdur:11s;--sdelay:-3s; --sop:0.50;"></div>
""", unsafe_allow_html=True)

# Helper function to apply high-aesthetic dark theme to Plotly charts
def apply_dark_theme(fig, title="", height=400):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#F8FAFC"),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(12, 16, 32, 0.45)',
        font=dict(family='Plus Jakarta Sans, sans-serif', size=11, color='#94A3B8'),
        margin=dict(l=40, r=25, t=46 if title else 20, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)',
            zerolinecolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(color='#94A3B8')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)',
            zerolinecolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(color='#94A3B8')
        ),
        hoverlabel=dict(
            bgcolor='rgba(12, 16, 32, 0.95)',
            bordercolor='rgba(167, 139, 250, 0.5)',
            font=dict(family='Plus Jakarta Sans, sans-serif', size=12, color='#FFFFFF')
        ),
        legend=dict(
            font=dict(color='#E2E8F0', size=11),
            bgcolor='rgba(12, 16, 32, 0.6)',
            bordercolor='rgba(255, 255, 255, 0.08)'
        ),
        height=height
    )
    return fig

ROOT_DIR = Path(__file__).resolve().parent.parent

# Load data helper functions
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(ROOT_DIR / 'data/processed/weather_processed.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading processed data: {str(e)}")
        return None

@st.cache_data
def load_featured_data():
    try:
        df = pd.read_csv(ROOT_DIR / 'data/processed/weather_featured.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading featured data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    try:
        temp_model = joblib.load(ROOT_DIR / 'models/temperature_model.pkl')
        rain_model = joblib.load(ROOT_DIR / 'models/rainfall_model.pkl')
        rain_encoder = joblib.load(ROOT_DIR / 'models/rainfall_encoder.pkl')
        temp_metrics = joblib.load(ROOT_DIR / 'models/temperature_metrics.pkl')
        rain_metrics = joblib.load(ROOT_DIR / 'models/rainfall_metrics.pkl')
        return temp_model, rain_model, rain_encoder, temp_metrics, rain_metrics
    except Exception as e:
        st.error(f"Error loading models/metrics: {str(e)}")
        return None, None, None, None, None

@st.cache_data(ttl=120)
def load_live_50_cities():
    """Cache live 50-city batch data for 2 minutes"""
    return fetch_all_50_cities_live()

# Load App Assets
df = load_data()
df_featured = load_featured_data()
temp_model, rain_model, rain_encoder, temp_metrics, rain_metrics = load_models()

# ==================== TOP NAVIGATION & HEADER ====================
top_col1, top_col2, top_col3 = st.columns([3, 2, 1])
with top_col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; padding: 2px 0;">
        {get_svg_icon("activity", "#A78BFA", 30)}
        <div>
            <div style="font-size: 1.65rem; font-weight: 800; color: #FFFFFF; line-height: 1.1;">
                ClimateAI <span class="logo-gradient">India</span>
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 2px;">
                Atmospheric &amp; Aerospace Meteorological Intelligence
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px; height: 100%;">
        <span class="badge-pill">
            <span class="pulse-dot"></span> LIVE TELEMETRY STREAM
        </span>
        <span style="font-size: 0.78rem; color: #64748B;">Synced: <b>{datetime.now().strftime("%H:%M:%S IST")}</b></span>
    </div>
    """, unsafe_allow_html=True)

with top_col3:
    if st.button("Refresh Telemetry", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Top Segmented Navigation Menu
PAGES = [
    "Overview Dashboard",
    "Live & 10-Day Forecast",
    "Aerospace & Turbulence",
    "Geospatial Radar Map",
    "Location Analysis",
    "Scenario Simulator",
    "Model Performance"
]

page = st.radio("NAVIGATION", PAGES, horizontal=True, label_visibility="collapsed")

# ----------------- SCREEN 1: OVERVIEW DASHBOARD -----------------
if page == "Overview Dashboard":
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <span class="badge-pill" style="margin-bottom: 8px;">
                    <span class="pulse-dot"></span> NATIONAL OBSERVATORY
                </span>
                <h1 class="main-header">National Climate &amp; Sensor Telemetry</h1>
                <p style="color: #94A3B8; font-size: 0.98rem; max-width: 680px; margin-top: 4px; line-height: 1.5;">
                    Continuous atmospheric sensor streams, decadal climatological dynamics, and multi-horizon AI hazard forecasting across 50 verified urban meteorological stations.
                </p>
            </div>
            <div style="background: rgba(12,16,32,0.85); border: 1px solid rgba(167,139,250,0.2); border-radius: 14px; padding: 14px 20px;">
                <div style="font-size: 0.7rem; color: #64748B; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">Validation Accuracy</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #A78BFA; margin-top: 2px;">0.68°C MAE</div>
                <div style="font-size: 0.76rem; color: #34D399; margin-top: 2px;">97.54% Variance Explained</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch live 50-city batch data
    with st.spinner("Streaming live 50-station meteorological telemetry..."):
        df_live_50 = load_live_50_cities()

    if not df_live_50.empty:
        # ── Live scrolling continuous ticker ──────────────────────────
        ticker_items = ""
        for _, row in df_live_50.iterrows():
            temp_color = "#F87171" if row['Temperature'] > 34 else ("#FBBF24" if row['Temperature'] > 28 else "#A78BFA")
            rain_color = "#34D399" if row['Rainfall'] > 3 else "#64748B"
            ticker_items += f"""
            <span class="ticker-item">
                <span class="ticker-label">{row['City']}</span>
                <span style="color:{temp_color};">{row['Temperature']:.1f}°C</span>
                <span style="color:#334155;">·</span>
                <span style="color:{rain_color};">{row['Rainfall']:.1f}mm</span>
            </span>"""
        double_items = ticker_items + ticker_items
        st.markdown(f"""
        <div class="ticker-wrapper">
            <div class="ticker-track">{double_items}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<p class="section-title">{get_svg_icon("activity", "#A78BFA", 18)} National Sensor Network Summary</p>', unsafe_allow_html=True)
        
        l1, l2, l3, l4 = st.columns(4)
        hottest_live = df_live_50.loc[df_live_50['Temperature'].idxmax()]
        wettest_live = df_live_50.loc[df_live_50['Rainfall'].idxmax()]
        avg_live_temp = df_live_50['Temperature'].mean()
        hazard_count = len(df_live_50[df_live_50['Hazard'] != 'Normal'])

        with l1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{get_svg_icon("thermometer", "#A78BFA", 14)} NATIONAL MEAN TEMP</div>
                <div class="metric-value">{avg_live_temp:.1f}°C</div>
                <div class="metric-subtitle">Across 50 active stations</div>
            </div>
            """, unsafe_allow_html=True)

        with l2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{get_svg_icon("thermometer", "#F87171", 14)} MAXIMUM RECORDED</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #F87171, #FB7185); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{hottest_live['Temperature']:.1f}°C</div>
                <div class="metric-subtitle"><b>{hottest_live['City']}</b> ({hottest_live['Condition']})</div>
            </div>
            """, unsafe_allow_html=True)

        with l3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{get_svg_icon("droplet", "#34D399", 14)} WETTEST STATION</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #34D399, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{wettest_live['Rainfall']:.1f} <span style="font-size: 1.1rem;">mm</span></div>
                <div class="metric-subtitle"><b>{wettest_live['City']}</b> ({wettest_live['Condition']})</div>
            </div>
            """, unsafe_allow_html=True)

        with l4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{get_svg_icon("shield-alert", "#FBBF24", 14)} ACTIVE ADVISORIES</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #FBBF24, #F59E0B); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{hazard_count} <span style="font-size: 1.1rem;">stations</span></div>
                <div class="metric-subtitle">Heatwave / precipitation alerts</div>
            </div>
            """, unsafe_allow_html=True)

        # 2D Live Station Leaderboard & Temperature Distribution
        col_grid1, col_grid2 = st.columns([3, 2])

        with col_grid1:
            st.markdown(f'<p class="section-title">{get_svg_icon("bar-chart", "#A78BFA", 18)} National Thermal Distribution (All 50 Stations)</p>', unsafe_allow_html=True)
            sorted_live = df_live_50.sort_values('Temperature', ascending=False).reset_index(drop=True)
            t_vals = sorted_live['Temperature']
            n = len(t_vals)

            # Color by RANK (0→1) so bars always get a full gradient even when
            # all temperatures are nearly identical on a given day.
            rank_norm = [i / max(n - 1, 1) for i in range(n)]  # 1.0 = hottest, 0.0 = coolest
            bar_colors = pc.sample_colorscale('Inferno', rank_norm)

            t_lo, t_hi = t_vals.min(), t_vals.max()

            fig_bar_live = go.Figure()
            fig_bar_live.add_trace(go.Bar(
                x=sorted_live['City'],
                y=t_vals,
                marker=dict(color=bar_colors),
                hovertemplate='<b>%{x}</b><br>Live Temp: %{y:.1f}°C<extra></extra>'
            ))
            # Invisible scatter for a proper colorbar legend
            fig_bar_live.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    colorscale='Inferno',
                    cmin=t_lo, cmax=t_hi,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Temp (°C)", font=dict(color="#CBD5E1", size=11)),
                        tickfont=dict(color="#CBD5E1", size=10),
                        bgcolor='rgba(12, 16, 32, 0.7)',
                        bordercolor='rgba(167, 139, 250, 0.2)',
                        thickness=12, len=0.75,
                        tickvals=[t_lo, (t_lo+t_hi)/2, t_hi],
                        ticktext=[f"{t_lo:.1f}°C", f"{(t_lo+t_hi)/2:.1f}°C", f"{t_hi:.1f}°C"]
                    ),
                    color=[t_lo, t_hi]
                ),
                showlegend=False
            ))
            apply_dark_theme(fig_bar_live, height=360)
            fig_bar_live.update_layout(xaxis=dict(tickangle=-65, showgrid=False, tickfont=dict(size=9, color='#CBD5E1')))
            st.plotly_chart(fig_bar_live, use_container_width=True)


        with col_grid2:
            st.markdown(f'<p class="section-title">{get_svg_icon("shield-alert", "#FBBF24", 18)} Station Extremes Leaderboard</p>', unsafe_allow_html=True)
            top_hot = df_live_50.nlargest(4, 'Temperature')[['City', 'Temperature', 'Humidity', 'Condition']]
            top_wet = df_live_50.nlargest(3, 'Rainfall')[['City', 'Rainfall', 'WindSpeed', 'Condition']]

            st.markdown("""
            <div class="glass-card" style="padding: 16px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #F87171; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Warmest Recorded Stations</div>
            """, unsafe_allow_html=True)
            for _, r in top_hot.iterrows():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.88rem;">
                    <span><b>{r['City']}</b> <span style="color: #64748B; font-size: 0.8rem;">({r['Condition']})</span></span>
                    <span style="font-weight: 700; color: #F87171;">{r['Temperature']:.1f}°C</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="font-size: 0.8rem; font-weight: 700; color: #34D399; margin-top: 14px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Precipitation Leaders</div>
            """, unsafe_allow_html=True)
            for _, r in top_wet.iterrows():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.88rem;">
                    <span><b>{r['City']}</b> <span style="color: #64748B; font-size: 0.8rem;">(Wind: {r['WindSpeed']:.0f} km/h)</span></span>
                    <span style="font-weight: 700; color: #34D399;">{r['Rainfall']:.1f} mm</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # 2D Decadal Climate Heatmap Matrix (Year vs Month)
    if df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<p class="section-title">{get_svg_icon("calendar", "#A78BFA", 18)} Decadal National Climatology Heatmap (2010–2019 Monthly Matrix)</p>', unsafe_allow_html=True)

        df_temp_matrix = df.copy()
        df_temp_matrix['Year'] = df_temp_matrix['Date'].dt.year
        df_temp_matrix['Month'] = df_temp_matrix['Date'].dt.month

        temp_pivot = df_temp_matrix.groupby(['Year', 'Month'])['Temperature'].mean().reset_index()
        temp_pivot_matrix = temp_pivot.pivot(index='Month', columns='Year', values='Temperature')

        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=temp_pivot_matrix.values,
            x=[str(y) for y in temp_pivot_matrix.columns],
            y=month_labels,
            colorscale='Plasma',
            xgap=3,
            ygap=3,
            colorbar=dict(
                title=dict(text="Mean Temp (°C)", font=dict(color="#CBD5E1", size=11)),
                tickfont=dict(color="#CBD5E1", size=10),
                bgcolor='rgba(12, 16, 32, 0.7)',
                bordercolor='rgba(167, 139, 250, 0.2)',
                borderwidth=1,
                thickness=14,
                len=0.75
            ),
            hovertemplate='<b>%{y} %{x}</b><br>National Mean Temp: <b>%{z:.2f}°C</b><extra></extra>'
        ))

        apply_dark_theme(fig_heat, height=370)
        fig_heat.update_layout(
            xaxis=dict(title="Calendar Year", showgrid=False, tickfont=dict(color='#CBD5E1')),
            yaxis=dict(title="Month of Year", showgrid=False, tickfont=dict(color='#CBD5E1'))
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown(f'<p class="section-title">{get_svg_icon("thermometer", "#A78BFA", 18)} Decadal Temperature Frequency</p>', unsafe_allow_html=True)
            temp_hist = df['Temperature'].value_counts(bins=35).sort_index()

            fig_dist = go.Figure(data=[go.Bar(
                x=[f"{interval.left:.0f}–{interval.right:.0f}°C" for interval in temp_hist.index],
                y=temp_hist.values,
                marker=dict(
                    color=temp_hist.values,
                    colorscale='Viridis',
                    line=dict(color='rgba(255,255,255,0.1)', width=1)
                ),
                hovertemplate='Range: %{x}<br>Recorded Days: %{y:,}<extra></extra>'
            )])
            apply_dark_theme(fig_dist, height=340)
            fig_dist.update_layout(xaxis=dict(tickangle=-45, showgrid=False, tickfont=dict(color='#CBD5E1')))
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_c2:
            st.markdown(f'<p class="section-title">{get_svg_icon("droplet", "#34D399", 18)} National Monthly Precipitation Profile</p>', unsafe_allow_html=True)

            monthly_rain_nat = df.groupby(df['Date'].dt.month)['Rainfall'].mean().reset_index()
            fig_rain_bar = go.Figure(data=[go.Bar(
                x=month_labels,
                y=monthly_rain_nat['Rainfall'],
                marker=dict(
                    color=monthly_rain_nat['Rainfall'],
                    colorscale='Teal',
                    line=dict(color='rgba(255,255,255,0.15)', width=1)
                ),
                hovertemplate='%{x}: %{y:.2f} mm/day<extra></extra>'
            )])
            apply_dark_theme(fig_rain_bar, height=340)
            fig_rain_bar.update_layout(yaxis_title="Mean Daily Rainfall (mm)")
            st.plotly_chart(fig_rain_bar, use_container_width=True)

# ----------------- SCREEN 2: LIVE & 10-DAY FORECAST -----------------
elif page == "Live & 10-Day Forecast":
    st.markdown(f'<p class="main-header">Live Telemetry &amp; 10-Day Multi-Horizon AI Forecast</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time sensor telemetry, today\'s full atmospheric breakdown, 24-hour progression, 10-day AI outlook, and calibrated XGBoost next-day prediction.</p>', unsafe_allow_html=True)
    st.markdown("---")

    if df_featured is not None:
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            cities = sorted(df_featured['City'].unique())
            selected_city = st.selectbox("Target Urban Meteorological Station", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
        with col_c2:
            forecast_mode = st.radio(
                "Data Stream Source",
                ["Live Real-Time Stream (Open-Meteo REST & AQI)", "Historical Archive Mode (2010–2025)"],
                horizontal=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if forecast_mode.startswith("Live"):
            with st.spinner(f"Querying live sensors and air quality telemetry for {selected_city}..."):
                live_res = predict_live_weather(selected_city)

            if live_res.get('status') == 'success':
                st.markdown(f'<p class="section-title">{get_svg_icon("activity", "#A78BFA", 18)} Today\'s Real-Time Sensor Snapshot • {selected_city} ({live_res.get("condition")})</p>', unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                aqi = live_res.get('air_quality', {})

                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{get_svg_icon("thermometer", "#A78BFA", 14)} AMBIENT TEMPERATURE</div>
                        <div class="metric-value">{live_res['current_temp']:.1f}°C</div>
                        <div class="metric-subtitle">Feels Like: <b>{live_res.get('apparent_temp', live_res['current_temp']):.1f}°C</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    aqi_val = aqi.get('us_aqi', 65)
                    aqi_cat = aqi.get('category', 'Moderate')
                    aqi_col = '#34D399' if aqi_val <= 50 else ('#FBBF24' if aqi_val <= 100 else '#F87171')
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{get_svg_icon("activity", aqi_col, 14)} AIR QUALITY INDEX</div>
                        <div class="metric-value" style="color: {aqi_col};">{aqi_val} <span style="font-size: 1.1rem;">AQI</span></div>
                        <div class="metric-subtitle">PM2.5: {aqi.get('pm2_5', 20):.1f} | Category: <b>{aqi_cat}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{get_svg_icon("droplet", "#34D399", 14)} RELATIVE HUMIDITY</div>
                        <div class="metric-value" style="background: linear-gradient(135deg, #34D399, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{live_res['current_humidity']}%</div>
                        <div class="metric-subtitle">Dew Point: {live_res.get('dew_point', 18):.1f}°C | Press: {live_res.get('current_pressure', 1012):.0f} hPa</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{get_svg_icon("wind", "#C084FC", 14)} WIND SPEED</div>
                        <div class="metric-value">{live_res['current_wind']:.1f} <span style="font-size: 1.1rem;">km/h</span></div>
                        <div class="metric-subtitle">GPS: {live_res.get('latitude', 28.6):.2f}°N, {live_res.get('longitude', 77.2):.2f}°E</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # AI Predictions for Tomorrow
                st.markdown(f'<p class="section-title">{get_svg_icon("cpu", "#A78BFA", 18)} Calibrated AI Forecast for Tomorrow (XGBoost)</p>', unsafe_allow_html=True)
                col_p1, col_p2 = st.columns(2)

                pred_temp = live_res['ai_next_day_temp']
                pred_rain = live_res['ai_next_day_rain']

                with col_p1:
                    city_hist = df[df['City'] == selected_city] if df is not None else None
                    hist_temp_avg = city_hist['Temperature'].mean() if city_hist is not None else 25.0
                    temp_badge_color = '#F87171' if pred_temp > 35 else ('#FBBF24' if pred_temp > 28 else '#34D399')

                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 4px solid {temp_badge_color};">
                        <div class="metric-title">CALIBRATED TEMPERATURE (TOMORROW)</div>
                        <div style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 8px 0;">
                            {pred_temp:.1f}°C
                        </div>
                        <div style="color: #94A3B8; font-size: 0.92rem;">
                            10-Yr Historical Baseline: <b>{hist_temp_avg:.1f}°C</b> (Departure: <span style="color: {'#F87171' if pred_temp > hist_temp_avg else '#A78BFA'}">{pred_temp - hist_temp_avg:+.1f}°C</span>)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_p2:
                    rain_badge_color = '#34D399' if pred_rain == 'No Rain' else ('#A78BFA' if pred_rain == 'Light Rain' else ('#818CF8' if pred_rain == 'Moderate Rain' else '#F87171'))
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 4px solid {rain_badge_color}; background: linear-gradient(135deg, rgba(167, 139, 250, 0.1) 0%, rgba(12, 16, 32, 0.85) 100%);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="metric-title">PRECIPITATION CLASSIFICATION</div>
                            <span class="badge-pill" style="background: rgba(167, 139, 250, 0.15); border-color: {rain_badge_color}; color: {rain_badge_color};">{pred_rain}</span>
                        </div>
                        <div style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 8px 0;">
                            {pred_rain}
                        </div>
                        <div style="color: #94A3B8; font-size: 0.92rem;">
                            Calibrated 4-Tier Classifier (Zero / Light / Moderate / Convective)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Active Hazard Advisory Cards
                if pred_temp >= 40.0:
                    st.markdown(f"""
                    <div class="alert-card alert-heatwave">
                        {get_svg_icon("shield-alert", "#F87171", 24)}
                        <div>
                            <b>HEATWAVE WARNING:</b> Forecasted temperature ({pred_temp:.1f}°C) exceeds extreme threshold. Hydration advisories and solar radiation alerts active.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_rain == "Heavy Rain":
                    st.markdown(f"""
                    <div class="alert-card alert-rain">
                        {get_svg_icon("shield-alert", "#34D399", 24)}
                        <div>
                            <b>HEAVY RAINFALL ADVISORY:</b> Intense precipitation forecasted. Low-lying areas should prepare for waterlogging and traffic delays.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-card alert-normal">
                        {get_svg_icon("check-circle", "#34D399", 24)}
                        <div>
                            <b>ATMOSPHERIC CONDITIONS NOMINAL:</b> Weather parameters remain within safe seasonal baseline limits for {selected_city}.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 24-Hour Continuous Hourly Progression
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<p class="section-title">{get_svg_icon("activity", "#A78BFA", 18)} 24-Hour Hourly Atmospheric Progression • {selected_city}</p>', unsafe_allow_html=True)
                
                h_df = pd.DataFrame(live_res.get('hourly_24h', []))
                if not h_df.empty:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(
                        x=h_df['time'],
                        y=h_df['temp'],
                        name="Temperature (°C)",
                        mode='lines+markers',
                        line=dict(color='#A78BFA', width=3, shape='linear'),
                        marker=dict(size=7, color='#C084FC', line=dict(color='white', width=1)),
                        hovertemplate='<b>Time: %{x}</b><br>Temperature: <b>%{y:.1f}°C</b><extra></extra>'
                    ))
                    fig_h.add_trace(go.Bar(
                        x=h_df['time'],
                        y=h_df['rain_prob'],
                        name="Rain Probability (%)",
                        yaxis='y2',
                        marker=dict(color='rgba(52, 211, 153, 0.45)', line=dict(color='#34D399', width=1)),
                        hovertemplate='<b>Time: %{x}</b><br>Rain Probability: <b>%{y}%</b><extra></extra>'
                    ))
                    apply_dark_theme(fig_h, height=350)
                    fig_h.update_layout(
                        xaxis=dict(
                            title="Hour of Day (IST)",
                            showgrid=True,
                            tickangle=0,
                            tickfont=dict(color="#CBD5E1", size=10)
                        ),
                        yaxis=dict(
                            title="Temperature (°C)",
                            tickfont=dict(color="#CBD5E1"),
                            showgrid=True
                        ),
                        yaxis2=dict(
                            title="Precipitation Chance (%)",
                            overlaying='y',
                            side='right',
                            range=[0, 100],
                            showgrid=False,
                            tickfont=dict(color='#34D399')
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

                # 10-DAY MULTI-HORIZON AI OUTLOOK
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<p class="section-title">{get_svg_icon("calendar", "#A78BFA", 18)} 10-Day Multi-Horizon Meteorological Outlook • {selected_city}</p>', unsafe_allow_html=True)

                f_df = pd.DataFrame(live_res.get('forecast_10d', []))
                if not f_df.empty:
                    # Render 10 interactive cards
                    cols_10 = st.columns(min(len(f_df), 10))
                    for idx, r in f_df.iterrows():
                        if idx < len(cols_10):
                            with cols_10[idx]:
                                try:
                                    d_obj = datetime.strptime(r['date'], "%Y-%m-%d")
                                    day_str = d_obj.strftime("%a")
                                    date_str = d_obj.strftime("%d %b")
                                except Exception:
                                    day_str = f"D+{idx+1}"
                                    date_str = r['date']
                                
                                cond_color = "#34D399" if "Rain" in r['condition'] else ("#FBBF24" if "Clear" in r['condition'] or "Sunny" in r['condition'] else "#A78BFA")
                                st.markdown(f"""
                                <div class="day-card">
                                    <div style="font-size: 0.8rem; font-weight: 700; color: #A78BFA;">{day_str}</div>
                                    <div style="font-size: 0.72rem; color: #64748B; margin-bottom: 6px;">{date_str}</div>
                                    <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">{r['temp_max']:.0f}°</div>
                                    <div style="font-size: 0.85rem; color: #64748B;">{r['temp_min']:.0f}°</div>
                                    <div style="font-size: 0.72rem; color: {cond_color}; font-weight: 600; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{r['condition']}</div>
                                    <div style="font-size: 0.68rem; color: #34D399; margin-top: 2px;">{r.get('rain_prob', 0)}% rain</div>
                                </div>
                                """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    # 10-Day Temperature Envelope (Clean Categorical Multi-Day Range)
                    display_dates = []
                    for d_str in f_df['date']:
                        try:
                            display_dates.append(datetime.strptime(d_str, "%Y-%m-%d").strftime("%a, %d %b"))
                        except Exception:
                            display_dates.append(d_str)

                    fig_10d = go.Figure()
                    fig_10d.add_trace(go.Scatter(
                        x=display_dates,
                        y=f_df['temp_max'],
                        name="Max Temp (°C)",
                        mode='lines+markers+text',
                        text=[f"{v:.0f}°" for v in f_df['temp_max']],
                        textposition='top center',
                        textfont=dict(size=10, color='#F87171'),
                        line=dict(color='#F87171', width=3, shape='linear'),
                        marker=dict(size=8, color='#F87171', line=dict(color='white', width=1)),
                        hovertemplate='<b>%{x}</b><br>High: <b>%{y:.1f}°C</b><extra></extra>'
                    ))
                    fig_10d.add_trace(go.Scatter(
                        x=display_dates,
                        y=f_df['temp_min'],
                        name="Min Temp (°C)",
                        mode='lines+markers+text',
                        text=[f"{v:.0f}°" for v in f_df['temp_min']],
                        textposition='bottom center',
                        textfont=dict(size=10, color='#A78BFA'),
                        line=dict(color='#A78BFA', width=3, shape='linear'),
                        marker=dict(size=8, color='#A78BFA', line=dict(color='white', width=1)),
                        fill='tonexty',
                        fillcolor='rgba(167, 139, 250, 0.14)',
                        hovertemplate='<b>%{x}</b><br>Low: <b>%{y:.1f}°C</b><extra></extra>'
                    ))
                    apply_dark_theme(fig_10d, "10-Day Multi-Horizon Temperature Trajectory & Thermal Range", height=380)
                    fig_10d.update_layout(
                        xaxis=dict(
                            type='category',
                            showgrid=True,
                            tickfont=dict(color='#CBD5E1', size=10),
                            tickangle=0
                        ),
                        yaxis=dict(
                            title="Temperature (°C)",
                            showgrid=True,
                            range=[max(0, f_df['temp_min'].min() - 4), f_df['temp_max'].max() + 5]
                        )
                    )
                    st.plotly_chart(fig_10d, use_container_width=True)

        else:
            # ── HISTORICAL ARCHIVE VERIFICATION MODE (2010–2025) ─────
            st.markdown(f'<p class="section-title">{get_svg_icon("calendar", "#A78BFA", 18)} 15-Year Historical Climatological Engine &amp; AI Benchmark (2010–2025)</p>', unsafe_allow_html=True)
            
            city_all_hist = df_featured[df_featured['City'] == selected_city].sort_values('Date').copy()
            city_all_hist['Date_Only'] = city_all_hist['Date'].dt.date
            
            min_date = city_all_hist['Date_Only'].min()
            max_date = city_all_hist['Date_Only'].max()

            col_d1, col_d2 = st.columns([2, 2])
            with col_d1:
                target_date = st.date_input(
                    "Select Historical Verification Date (2010 to 2025)",
                    value=pd.to_datetime("2024-05-28").date() if max_date >= pd.to_datetime("2024-05-28").date() else max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            with col_d2:
                PRESET_EVENTS = {
                    "Select a Climatological Preset...": None,
                    "2024 Record Summer Heatwave (28 May 2024)": "2024-05-28",
                    "2023 Severe Monsoon Inundation (10 Jul 2023)": "2023-07-10",
                    "2022 Early Spring Heat Anomaly (20 Mar 2022)": "2022-03-20",
                    "2020 Clean Air & Low Thermal Deviation (15 Apr 2020)": "2020-04-15",
                    "2019 Summer Peak Heatwave (10 Jun 2019)": "2019-06-10",
                    "2018 Winter Minimum Chill (05 Jan 2018)": "2018-01-05",
                    "2015 Severe National Heatwave (25 May 2015)": "2015-05-25",
                    "2014 Super Monsoon Surge (28 Jul 2014)": "2014-07-28"
                }
                preset_choice = st.selectbox("Quick 15-Year Climate Event Presets", list(PRESET_EVENTS.keys()))
                if PRESET_EVENTS[preset_choice] is not None:
                    target_date = pd.to_datetime(PRESET_EVENTS[preset_choice]).date()

            # Find matching or closest historical record
            matching_obs = city_all_hist[city_all_hist['Date_Only'] == target_date]
            if matching_obs.empty:
                # Find nearest date
                closest_idx = (city_all_hist['Date'] - pd.to_datetime(target_date)).abs().idxmin()
                city_obs = city_all_hist.loc[closest_idx]
            else:
                city_obs = matching_obs.iloc[0]

            obs_date_dt = pd.to_datetime(city_obs['Date'])

            # Run model inference on the historical day
            X_temp_hist = pd.DataFrame([city_obs[temp_metrics['features']]])
            X_rain_hist = pd.DataFrame([city_obs[rain_metrics['features']]])

            pred_temp = float(temp_model.predict(X_temp_hist)[0])
            pred_rain_code = int(rain_model.predict(X_rain_hist)[0])
            pred_rain_label = str(rain_encoder.inverse_transform([pred_rain_code])[0])

            actual_temp = float(city_obs['Temperature'])
            actual_rain = float(city_obs['Rainfall'])
            temp_error = abs(pred_temp - actual_temp)

            # Historical baseline for that month
            hist_normal_temp = city_all_hist[city_all_hist['Month'] == city_obs['Month']]['Temperature'].mean()
            temp_departure = actual_temp - hist_normal_temp

            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)

            with k1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("cpu", "#A78BFA", 14)} AI PREDICTED TEMP</div>
                    <div class="metric-value">{pred_temp:.1f}°C</div>
                    <div class="metric-subtitle">Date: <b>{obs_date_dt.strftime('%d %b %Y')}</b></div>
                </div>
                """, unsafe_allow_html=True)

            with k2:
                err_col = '#34D399' if temp_error <= 0.8 else ('#FBBF24' if temp_error <= 1.8 else '#F87171')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("check-circle", err_col, 14)} ACTUAL OBSERVED TEMP</div>
                    <div class="metric-value" style="color: #FFFFFF;">{actual_temp:.1f}°C</div>
                    <div class="metric-subtitle">Residual Error: <b style="color:{err_col};">{temp_error:.2f}°C</b> (MAE: 0.68°C)</div>
                </div>
                """, unsafe_allow_html=True)

            with k3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("droplet", "#34D399", 14)} AI RAINFALL CLASS</div>
                    <div class="metric-value" style="color: #34D399; font-size: 1.6rem;">{pred_rain_label}</div>
                    <div class="metric-subtitle">Observed Rain: <b>{actual_rain:.1f} mm</b></div>
                </div>
                """, unsafe_allow_html=True)

            with k4:
                dep_col = '#F87171' if temp_departure > 1.5 else ('#34D399' if abs(temp_departure) <= 1.5 else '#A78BFA')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("thermometer", dep_col, 14)} MONTHLY DEPARTURE</div>
                    <div class="metric-value" style="color: {dep_col}; font-size: 1.8rem;">{temp_departure:+.1f}°C</div>
                    <div class="metric-subtitle">10-Yr Baseline: <b>{hist_normal_temp:.1f}°C</b></div>
                </div>
                """, unsafe_allow_html=True)

            # ── 15-Day Historical vs AI Model Trajectory Window ─────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<p class="section-title">{get_svg_icon("activity", "#A78BFA", 18)} 15-Day Verification Window (Observed vs AI Model vs 10-Yr Baseline)</p>', unsafe_allow_html=True)

            # Extract 7 days before and 7 days after
            target_idx = city_all_hist[city_all_hist['Date'] == city_obs['Date']].index
            if len(target_idx) > 0:
                pos = city_all_hist.index.get_loc(target_idx[0])
                start_pos = max(0, pos - 7)
                end_pos = min(len(city_all_hist), pos + 8)
                window_df = city_all_hist.iloc[start_pos:end_pos].copy()
            else:
                window_df = city_all_hist.head(15).copy()

            # Predict for all window points
            X_win = window_df[temp_metrics['features']]
            window_df['AI_Predicted_Temp'] = temp_model.predict(X_win)
            window_df['Display_Date'] = window_df['Date'].dt.strftime('%d %b (%a)')

            # Monthly baseline for each point
            window_df['Seasonal_Normal'] = window_df['Month'].apply(
                lambda m: city_all_hist[city_all_hist['Month'] == m]['Temperature'].mean()
            )

            fig_win = go.Figure()

            # 1. 10-Yr Climatological Normal Baseline
            fig_win.add_trace(go.Scatter(
                x=window_df['Display_Date'],
                y=window_df['Seasonal_Normal'],
                name='10-Yr Climatological Normal',
                mode='lines',
                line=dict(color='rgba(167, 139, 250, 0.4)', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>10-Yr Normal: %{y:.1f}°C<extra></extra>'
            ))

            # 2. Actual Ground Truth
            fig_win.add_trace(go.Scatter(
                x=window_df['Display_Date'],
                y=window_df['Temperature'],
                name='Actual Observed Ground Truth',
                mode='lines+markers',
                line=dict(color='#34D399', width=3, shape='linear'),
                marker=dict(size=8, color='#34D399', line=dict(color='white', width=1)),
                hovertemplate='Date: %{x}<br>Actual Observed: <b>%{y:.1f}°C</b><extra></extra>'
            ))

            # 3. AI Model Predictions
            fig_win.add_trace(go.Scatter(
                x=window_df['Display_Date'],
                y=window_df['AI_Predicted_Temp'],
                name='XGBoost AI Model Prediction',
                mode='lines+markers',
                line=dict(color='#C084FC', width=3, shape='linear'),
                marker=dict(size=8, color='#C084FC', symbol='diamond', line=dict(color='white', width=1)),
                hovertemplate='Date: %{x}<br>AI Predicted: <b>%{y:.1f}°C</b><extra></extra>'
            ))

            # 4. Highlight the selected target date
            sel_display = obs_date_dt.strftime('%d %b (%a)')
            if sel_display in window_df['Display_Date'].values:
                fig_win.add_trace(go.Scatter(
                    x=[sel_display],
                    y=[actual_temp],
                    name='Selected Target Day',
                    mode='markers',
                    marker=dict(size=16, color='#F87171', symbol='circle-open', line=dict(color='#F87171', width=3)),
                    hovertemplate='<b>Target Verification Day</b><br>Observed: %{y:.1f}°C<extra></extra>'
                ))

            apply_dark_theme(fig_win, f"15-Day Thermal Trajectory Benchmark • {selected_city} ({obs_date_dt.strftime('%B %Y')})", height=380)
            fig_win.update_layout(
                xaxis=dict(type='category', showgrid=True, tickfont=dict(color='#CBD5E1', size=10)),
                yaxis=dict(title="Temperature (°C)", showgrid=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_win, use_container_width=True)

            # ── Residual Error & Moisture Breakdown ─────
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                # Daily Prediction Residual (T_pred - T_obs)
                window_df['Residual_Error'] = window_df['AI_Predicted_Temp'] - window_df['Temperature']
                fig_res = go.Figure(data=[go.Bar(
                    x=window_df['Display_Date'],
                    y=window_df['Residual_Error'],
                    marker=dict(
                        color=['#34D399' if abs(e) <= 0.8 else ('#FBBF24' if abs(e) <= 1.8 else '#F87171') for e in window_df['Residual_Error']],
                        line=dict(color='rgba(255,255,255,0.15)', width=1)
                    ),
                    hovertemplate='Date: %{x}<br>Model Residual Error: <b>%{y:+.2f}°C</b><extra></extra>'
                )])
                apply_dark_theme(fig_res, "Daily AI Model Prediction Residual Error (T_pred − T_obs)", height=320)
                fig_res.update_layout(
                    xaxis=dict(type='category', showgrid=False, tickfont=dict(size=9, color='#CBD5E1')),
                    yaxis=dict(title="Error Residual (°C)", showgrid=True)
                )
                st.plotly_chart(fig_res, use_container_width=True)

            with col_g2:
                # Daily Rainfall in verification window
                fig_hist_rain = go.Figure(data=[go.Bar(
                    x=window_df['Display_Date'],
                    y=window_df['Rainfall'],
                    marker=dict(
                        color=window_df['Rainfall'],
                        colorscale='Teal',
                        line=dict(color='rgba(255,255,255,0.15)', width=1)
                    ),
                    hovertemplate='Date: %{x}<br>Observed Daily Rainfall: <b>%{y:.1f} mm</b><extra></extra>'
                )])
                apply_dark_theme(fig_hist_rain, "Historical Daily Precipitation in Window (mm)", height=320)
                fig_hist_rain.update_layout(
                    xaxis=dict(type='category', showgrid=False, tickfont=dict(size=9, color='#CBD5E1')),
                    yaxis=dict(title="Rainfall (mm)", showgrid=True)
                )
                st.plotly_chart(fig_hist_rain, use_container_width=True)

# ----------------- SCREEN 3: AEROSPACE & TURBULENCE -----------------
elif page == "Aerospace & Turbulence":
    st.markdown(f'<p class="main-header">Aerospace &amp; Aviation Meteorology Hub</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Clear-Air Turbulence (CAT) severity indices, boundary layer wind shear, flight level density altitude, and interactive corridor risk analytics.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Airplane fly-across background animation (only on this tab) ──────────
    st.markdown("""
<style>
  .plane-wrap {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
  }
  /* Each plane slides right, with a gentle sine-wave turbulence wobble */
  @keyframes flyAcross1 {
    0%   { transform: translateX(-120px) translateY(0px)      rotate(-2deg); opacity: 0; }
    4%   { opacity: 0.85; }
    50%  { transform: translateX(52vw)   translateY(-18px)    rotate(1deg);  opacity: 0.85; }
    96%  { opacity: 0.7; }
    100% { transform: translateX(110vw)  translateY(6px)      rotate(-1deg); opacity: 0; }
  }
  @keyframes flyAcross2 {
    0%   { transform: translateX(-120px) translateY(0px)      rotate(1deg);  opacity: 0; }
    5%   { opacity: 0.75; }
    40%  { transform: translateX(40vw)   translateY(22px)     rotate(-2deg); opacity: 0.75; }
    80%  { transform: translateX(80vw)   translateY(-10px)    rotate(2deg);  opacity: 0.70; }
    100% { transform: translateX(110vw)  translateY(5px)      rotate(-1deg); opacity: 0; }
  }
  @keyframes flyAcross3 {
    0%   { transform: translateX(-120px) translateY(0px)      rotate(-1deg); opacity: 0; }
    6%   { opacity: 0.60; }
    33%  { transform: translateX(33vw)   translateY(14px)     rotate(2deg);  opacity: 0.60; }
    66%  { transform: translateX(66vw)   translateY(-20px)    rotate(-2deg); opacity: 0.55; }
    100% { transform: translateX(110vw)  translateY(8px)      rotate(1deg);  opacity: 0; }
  }
  /* Contrail trail behind each plane */
  .plane-trail {
    position: absolute;
    height: 1.5px;
    width: 0;
    right: 100%;
    top: 50%;
    transform: translateY(-50%);
    background: linear-gradient(to left, rgba(255,255,255,0.45), transparent);
    animation: trailGrow var(--trail-dur, 4s) ease-out infinite;
  }
  @keyframes trailGrow {
    0%   { width: 0;    opacity: 0; }
    20%  { width: 80px; opacity: 0.5; }
    80%  { width: 60px; opacity: 0.3; }
    100% { width: 0;    opacity: 0; }
  }
  .plane {
    position: fixed;
    font-size: var(--plane-size, 22px);
    white-space: nowrap;
    filter: drop-shadow(0 0 6px rgba(167,139,250,0.55));
    line-height: 1;
  }
</style>

<div class="plane-wrap">
  <!-- Plane 1: high altitude, slow, large -->
  <div class="plane" style="top:12vh; --plane-size:28px;
       animation: flyAcross1 22s linear 0s infinite;">
    <div class="plane-trail" style="--trail-dur:22s;"></div>✈
  </div>
  <!-- Plane 2: mid altitude, medium speed -->
  <div class="plane" style="top:28vh; --plane-size:20px;
       animation: flyAcross2 16s linear -6s infinite; filter:drop-shadow(0 0 5px rgba(52,211,153,0.5));">
    <div class="plane-trail" style="--trail-dur:16s;"></div>✈
  </div>
  <!-- Plane 3: low altitude, fast, small -->
  <div class="plane" style="top:48vh; --plane-size:15px;
       animation: flyAcross3 11s linear -3s infinite; filter:drop-shadow(0 0 4px rgba(251,191,36,0.5));">
    <div class="plane-trail" style="--trail-dur:11s;"></div>✈
  </div>
  <!-- Plane 4: very high, delayed -->
  <div class="plane" style="top:7vh; --plane-size:18px;
       animation: flyAcross1 28s linear -12s infinite; filter:drop-shadow(0 0 5px rgba(192,132,252,0.45)); opacity:0.65;">
    <div class="plane-trail" style="--trail-dur:28s;"></div>✈
  </div>
  <!-- Plane 5: second pass at mid-low -->
  <div class="plane" style="top:38vh; --plane-size:13px;
       animation: flyAcross2 19s linear -9s infinite; filter:drop-shadow(0 0 4px rgba(248,113,113,0.45)); opacity:0.55;">
    <div class="plane-trail" style="--trail-dur:19s;"></div>✈
  </div>
  <!-- Plane 6: near ground level, tiny -->
  <div class="plane" style="top:62vh; --plane-size:11px;
       animation: flyAcross3 13s linear -5s infinite; opacity:0.45;">
    <div class="plane-trail" style="--trail-dur:13s;"></div>✈
  </div>
</div>
""", unsafe_allow_html=True)
    # ─────────────────────────────────────────────────────────────────────────

    # Fetch live 50 cities for aviation metrics
    with st.spinner("Calculating national aerospace and turbulence indices..."):
        df_live_50 = load_live_50_cities()

    if not df_live_50.empty:
        # Station & Route Selectors
        col_a1, col_a2 = st.columns([1, 2])
        with col_a1:
            airport_cities = sorted(df_live_50['City'].unique())
            selected_airport = st.selectbox("Target Aerodrome / Met Station", airport_cities, index=airport_cities.index("Mumbai") if "Mumbai" in airport_cities else 0)
        with col_a2:
            FLIGHT_ROUTES = [
                "DEL ✈ BOM (Delhi Indira Gandhi → Mumbai Chhatrapati Shivaji)",
                "BLR ✈ DEL (Bangalore Kempegowda → Delhi Indira Gandhi)",
                "BOM ✈ GOI (Mumbai → Goa Dabolim / Mopa)",
                "MAA ✈ CCU (Chennai → Kolkata Netaji Subhas)",
                "DEL ✈ SXR (Delhi → Srinagar - Himalayan Wave Corridor)",
                "HYD ✈ BOM (Hyderabad Rajiv Gandhi → Mumbai)"
            ]
            selected_route = st.selectbox("Aviation Flight Corridor Analyzer", FLIGHT_ROUTES)

        # Compute Aviation Telemetry for selected airport
        apt_row = df_live_50[df_live_50['City'] == selected_airport].iloc[0]
        aero_metrics = compute_aerospace_risk(
            temp=apt_row['Temperature'],
            pressure=apt_row['Pressure'],
            wind_speed=apt_row['WindSpeed'],
            humidity=apt_row['Humidity'],
            precip=apt_row['Rainfall']
        )

        st.markdown(f'<p class="section-title">{get_svg_icon("plane", "#A78BFA", 18)} Flight Level Atmospheric Indices • {selected_airport}</p>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{get_svg_icon("activity", aero_metrics['cat_color'], 14)} CAT TURBULENCE INDEX</div>
                <div class="metric-value" style="color: {aero_metrics['cat_color']};">{aero_metrics['cat_score']} <span style="font-size: 1.1rem;">/ 100</span></div>
                <div class="metric-subtitle">Level: <b>{aero_metrics['cat_category']}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{get_svg_icon("gauge", "#A78BFA", 14)} DENSITY ALTITUDE</div>
                <div class="metric-value">{aero_metrics['density_altitude_ft']:,.0f} <span style="font-size: 1.1rem;">ft</span></div>
                <div class="metric-subtitle">Press Alt: {aero_metrics['pressure_altitude_ft']:,.0f} ft | ISA Dev: {apt_row['Temperature']-15:+.1f}°C</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{get_svg_icon("wind", "#34D399", 14)} WIND SHEAR GRADIENT</div>
                <div class="metric-value">{aero_metrics['wind_shear_kt']} <span style="font-size: 1.1rem;">kt/kft</span></div>
                <div class="metric-subtitle">Surface Wind: {apt_row['WindSpeed']:.0f} km/h</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{get_svg_icon("sliders", "#FBBF24", 14)} TAKEOFF ROLL PENALTY</div>
                <div class="metric-value">+{aero_metrics['takeoff_penalty_pct']}%</div>
                <div class="metric-subtitle">Icing Risk: <b>{aero_metrics['icing_score']:.0f}%</b> at FL180</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Interactive Flight Corridor Cross-Section Simulation ─────
        st.markdown(f'<p class="section-title">{get_svg_icon("plane", "#A78BFA", 18)} En-Route Flight Corridor Turbulence Cross-Section ({selected_route.split(" ")[0]} ✈ {selected_route.split(" ")[2]})</p>', unsafe_allow_html=True)

        # Generate route altitude profile (FL100 to FL390 across 12 waypoint checkpoints)
        waypoints = np.linspace(0, 1200, 25)
        flight_alt = np.sin(np.linspace(0, np.pi, 25)) * 36000 + 4000
        
        # Simulate turbulence field along the corridor
        base_cat = aero_metrics['cat_score']
        corridor_cat = np.clip(base_cat + np.sin(waypoints/80) * 20 + np.random.normal(0, 5, 25), 5, 95)

        fig_route = go.Figure()

        # Add turbulence heat band along the route
        fig_route.add_trace(go.Scatter(
            x=waypoints,
            y=flight_alt,
            mode='lines+markers',
            name='Cruise Trajectory (FL360)',
            line=dict(color='#FFFFFF', width=3),
            marker=dict(
                size=10,
                color=corridor_cat,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title="CAT Risk (%)",
                    tickfont=dict(color="#CBD5E1", size=10),
                    bgcolor='rgba(12, 16, 32, 0.7)',
                    bordercolor='rgba(167, 139, 250, 0.2)',
                    thickness=12,
                    len=0.75
                )
            ),
            hovertemplate='Distance: %{x:.0f} km<br>Flight Level: FL%{y:.0f}<br>Turbulence Risk: %{marker.color:.1f}%<extra></extra>'
        ))

        # Safe cruise corridor upper/lower bounds
        fig_route.add_trace(go.Scatter(
            x=waypoints,
            y=[39000]*len(waypoints),
            mode='lines',
            name='Service Ceiling (FL390)',
            line=dict(color='rgba(167, 139, 250, 0.3)', dash='dash')
        ))
        fig_route.add_trace(go.Scatter(
            x=waypoints,
            y=[28000]*len(waypoints),
            mode='lines',
            name='Min Cruise Level (FL280)',
            line=dict(color='rgba(52, 211, 153, 0.3)', dash='dash')
        ))

        apply_dark_theme(fig_route, height=360)
        fig_route.update_layout(
            xaxis=dict(title="Corridor Distance from Departure (km)", showgrid=True),
            yaxis=dict(title="Altitude (ft MSL)", range=[0, 42000], showgrid=True)
        )
        st.plotly_chart(fig_route, use_container_width=True)

        # National Airport Aviation Risk Leaderboard
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<p class="section-title">{get_svg_icon("bar-chart", "#A78BFA", 18)} National Airport Aerospace Risk Leaderboard (All 50 Stations)</p>', unsafe_allow_html=True)

        cat_list = []
        for _, r in df_live_50.iterrows():
            m = compute_aerospace_risk(r['Temperature'], r['Pressure'], r['WindSpeed'], r['Humidity'], r['Rainfall'])
            cat_list.append({
                'City': r['City'],
                'CAT_Score': m['cat_score'],
                'Category': m['cat_category'],
                'Density_Alt': m['density_altitude_ft'],
                'Wind_Shear': m['wind_shear_kt'],
                'Takeoff_Penalty': m['takeoff_penalty_pct'],
                'Temp': r['Temperature'],
                'Wind': r['WindSpeed']
            })
        df_cat = pd.DataFrame(cat_list).sort_values('CAT_Score', ascending=False)

        col_t1, col_t2 = st.columns([3, 2])
        with col_t1:
            fig_cat_bar = go.Figure(data=[go.Bar(
                x=df_cat['City'][:25],
                y=df_cat['CAT_Score'][:25],
                marker=dict(
                    color=df_cat['CAT_Score'][:25],
                    colorscale='Plasma',
                    showscale=False
                ),
                hovertemplate='<b>%{x}</b><br>CAT Index: %{y:.1f}%<extra></extra>'
            )])
            apply_dark_theme(fig_cat_bar, "Top 25 Air Corridors by Turbulence Risk", height=360)
            fig_cat_bar.update_layout(xaxis=dict(tickangle=-65, showgrid=False, tickfont=dict(size=9, color='#CBD5E1')))
            st.plotly_chart(fig_cat_bar, use_container_width=True)

        with col_t2:
            rows_html = ""
            for _, r in df_cat.head(8).iterrows():
                risk_col = '#F87171' if r['CAT_Score'] > 60 else ('#FBBF24' if r['CAT_Score'] > 40 else '#34D399')
                city_name = str(r['City'])
                cat_label = str(r['Category']).split(' ')[0]
                density_alt = f"{r['Density_Alt']:,.0f}"
                wind_speed = f"{r['Wind']:.0f}"
                cat_score = f"{r['CAT_Score']:.1f}"
                rows_html += (
                    '<div style="display:flex;justify-content:space-between;align-items:center;'
                    'padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.07);">'
                    '<div>'
                    f'<div style="font-size:0.9rem;font-weight:700;color:#FFFFFF;">{city_name}</div>'
                    f'<div style="font-size:0.73rem;color:#94A3B8;margin-top:2px;">Alt: {density_alt} ft &nbsp;|&nbsp; Wind: {wind_speed} km/h</div>'
                    '</div>'
                    '<div style="text-align:right;">'
                    f'<div style="font-size:1.05rem;font-weight:800;color:{risk_col};">{cat_score}%</div>'
                    f'<div style="font-size:0.72rem;font-weight:600;color:{risk_col};">{cat_label}</div>'
                    '</div>'
                    '</div>'
                )

            card_html = (
                '<div style="background:rgba(12,16,32,0.75);backdrop-filter:blur(20px);'
                'border:1px solid rgba(167,139,250,0.15);border-radius:16px;padding:20px;">'
                '<div style="font-size:0.78rem;font-weight:700;color:#A78BFA;'
                'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:14px;">'
                '✈ Aviation Risk Matrix'
                '</div>'
                + rows_html +
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

# ----------------- SCREEN 4: GEOSPATIAL RADAR MAP -----------------
elif page == "Geospatial Radar Map":
    st.markdown(f'<p class="main-header">Atmospheric Doppler &amp; Thermal Radar Surface</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">High-resolution continuous density radar surfaces, thermal isobar contours, and vector wind overlays across the Indian subcontinent.</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Constructing realistic continuous meteorological radar map..."):
        df_live_50 = load_live_50_cities()

    if not df_live_50.empty:
        city_valid = df_live_50.dropna(subset=['Latitude', 'Longitude'])

        map_layer = st.radio(
            "Active Doppler Radar Layer",
            [
                "Atmospheric Thermal Density Contour (°C)",
                "Precipitation Doppler Radar Simulation (mm)",
                "Aviation Turbulence Risk Surface (CAT %)"
            ],
            horizontal=True
        )

        # Compute dynamic CAT score for the radar surface
        city_valid['CAT_Score'] = city_valid.apply(
            lambda r: compute_aerospace_risk(r['Temperature'], r['Pressure'], r['WindSpeed'], r['Humidity'], r['Rainfall'])['cat_score'],
            axis=1
        )

        if "Thermal" in map_layer:
            metric_col = 'Temperature'
            colorscale = 'Plasma'
            cbar_title = 'Live Temp (°C)'
            hover_label = 'Temperature'
            hover_unit = '°C'
            density_radius = 55
        elif "Precipitation" in map_layer:
            metric_col = 'Rainfall'
            colorscale = 'Teal'
            cbar_title = 'Rainfall (mm)'
            hover_label = 'Precipitation'
            hover_unit = ' mm'
            density_radius = 48
        else:
            metric_col = 'CAT_Score'
            colorscale = 'Hot_r'
            cbar_title = 'Turbulence Risk (%)'
            hover_label = 'CAT Risk'
            hover_unit = '%'
            density_radius = 52

        # Cross-version Plotly map traces (Plotly v6 go.Densitymap / v5 go.Densitymapbox)
        DensityTrace = getattr(go, 'Densitymap', getattr(go, 'Densitymapbox', None))
        ScatterTrace = getattr(go, 'Scattermap', getattr(go, 'Scattermapbox', None))

        fig_map = go.Figure()

        # 1. Continuous smooth Doppler / thermal density surface
        fig_map.add_trace(DensityTrace(
            lat=city_valid['Latitude'],
            lon=city_valid['Longitude'],
            z=city_valid[metric_col],
            radius=density_radius,
            colorscale=colorscale,
            opacity=0.72,
            showscale=True,
            colorbar=dict(
                title=dict(text=cbar_title, font=dict(color='#CBD5E1', size=11)),
                tickfont=dict(color='#CBD5E1', size=10),
                bgcolor='rgba(12, 16, 32, 0.75)',
                bordercolor='rgba(167, 139, 250, 0.25)',
                borderwidth=1,
                thickness=14,
                len=0.75
            )
        ))

        # 2. Sleek Glowing Station Beacon Nodes (no static cluttered text labels)
        fig_map.add_trace(ScatterTrace(
            lat=city_valid['Latitude'],
            lon=city_valid['Longitude'],
            mode='markers',
            text=city_valid['City'],
            customdata=city_valid[['Temperature', 'Humidity', 'WindSpeed', 'Condition', 'CAT_Score']].values,
            marker=dict(
                size=8,
                color='#FFFFFF',
                opacity=0.9
            ),
            hovertemplate=(
                '<b>%{text} Station</b><br>'
                'Live Temp: <b>%{customdata[0]:.1f}°C</b><br>'
                'Condition: %{customdata[3]}<br>'
                'Humidity: %{customdata[1]}% | Wind: %{customdata[2]:.0f} km/h<br>'
                'CAT Turbulence Index: %{customdata[4]:.1f}%'
                '<extra></extra>'
            )
        ))

        # 3. Optional Flight Corridor Vectors if Aerospace mode
        if "Turbulence" in map_layer:
            routes = [
                ((28.6139, 77.2090), (19.0760, 72.8777)), # DEL - BOM
                ((12.9716, 77.5946), (28.6139, 77.2090)), # BLR - DEL
                ((13.0827, 80.2707), (22.5726, 88.3639)), # MAA - CCU
                ((28.6139, 77.2090), (34.0837, 74.7973)), # DEL - SXR
            ]
            for r_start, r_end in routes:
                fig_map.add_trace(ScatterTrace(
                    lat=[r_start[0], r_end[0]],
                    lon=[r_start[1], r_end[1]],
                    mode='lines',
                    line=dict(width=2, color='rgba(167, 139, 250, 0.6)'),
                    hoverinfo='none',
                    showlegend=False
                ))

        map_config = dict(
            style="carto-darkmatter",
            center=dict(lat=22.5, lon=79.5),
            zoom=4.2
        )
        if hasattr(go, 'Densitymap'):
            fig_map.update_layout(map=map_config)
        else:
            fig_map.update_layout(mapbox=map_config)

        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<p class="section-title">{get_svg_icon("globe", "#A78BFA", 18)} Geographic Telemetry Span</p>', unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">TOTAL ACTIVE RADAR STATIONS</div>
                <div class="metric-value">{len(city_valid)}</div>
                <div class="metric-subtitle">Across all major states & union territories</div>
            </div>
            """, unsafe_allow_html=True)
        with g2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">LATITUDE COVERAGE</div>
                <div class="metric-value" style="font-size: 1.8rem;">{city_valid['Latitude'].min():.1f}°N – {city_valid['Latitude'].max():.1f}°N</div>
                <div class="metric-subtitle">Kochi to Srinagar</div>
            </div>
            """, unsafe_allow_html=True)
        with g3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">LONGITUDE COVERAGE</div>
                <div class="metric-value" style="font-size: 1.8rem;">{city_valid['Longitude'].min():.1f}°E – {city_valid['Longitude'].max():.1f}°E</div>
                <div class="metric-subtitle">Rajkot to Guwahati</div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- SCREEN 5: LOCATION ANALYSIS -----------------
elif page == "Location Analysis":
    st.markdown(f'<p class="main-header">Station Location Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Decadal trends, seasonal climatology, and station-to-station comparative analytics.</p>', unsafe_allow_html=True)
    st.markdown("---")

    if df is not None:
        cities = sorted(df['City'].unique())
        selected_city = st.selectbox("Select Urban Station", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)

        city_data = df[df['City'] == selected_city].copy()
        city_data['Year'] = city_data['Date'].dt.year
        city_data['Month'] = city_data['Date'].dt.month

        tab1, tab2, tab3 = st.tabs(["Thermal Climatology", "Precipitation Dynamics", "Comparative Station Benchmark"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("thermometer", "#A78BFA", 14)} MEAN TEMPERATURE</div>
                    <div class="metric-value">{city_data['Temperature'].mean():.1f}°C</div>
                    <div class="metric-subtitle">Decadal baseline for {selected_city}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("thermometer", "#F87171", 14)} RECORD MAXIMUM</div>
                    <div class="metric-value" style="background: linear-gradient(135deg, #F87171, #FB7185); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{city_data['Temperature'].max():.1f}°C</div>
                    <div class="metric-subtitle">Summer peak recorded</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("thermometer", "#A78BFA", 14)} RECORD MINIMUM</div>
                    <div class="metric-value" style="background: linear-gradient(135deg, #A78BFA, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{city_data['Temperature'].min():.1f}°C</div>
                    <div class="metric-subtitle">Winter low recorded</div>
                </div>
                """, unsafe_allow_html=True)

            yearly_avg_temp = city_data.groupby('Year')['Temperature'].mean().reset_index()
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=yearly_avg_temp['Year'],
                y=yearly_avg_temp['Temperature'],
                mode='lines+markers',
                line=dict(color='#A78BFA', width=3, shape='spline'),
                marker=dict(size=8, color='#C084FC', line=dict(color='white', width=1.5)),
                fill='tozeroy',
                fillcolor='rgba(167, 139, 250, 0.12)',
                hovertemplate='Year %{x}: %{y:.2f}°C<extra></extra>'
            ))
            apply_dark_theme(fig1, f"Annual Mean Temperature Trajectory • {selected_city}", height=360)
            st.plotly_chart(fig1, use_container_width=True)

            # 2D Monthly Climatology Heatmap for the city
            st.markdown(f'<p class="section-title">{get_svg_icon("calendar", "#A78BFA", 18)} Monthly Thermal Matrix (2010–2019) • {selected_city}</p>', unsafe_allow_html=True)
            temp_city_pivot = city_data.groupby(['Year', 'Month'])['Temperature'].mean().reset_index()
            temp_city_matrix = temp_city_pivot.pivot(index='Month', columns='Year', values='Temperature')

            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            fig_city_heat = go.Figure(data=go.Heatmap(
                z=temp_city_matrix.values,
                x=[str(y) for y in temp_city_matrix.columns],
                y=month_names,
                colorscale='Plasma',
                xgap=3,
                ygap=3,
                colorbar=dict(
                    title=dict(text="Temp (°C)", font=dict(color="#CBD5E1", size=11)),
                    tickfont=dict(color="#CBD5E1", size=10),
                    bgcolor='rgba(12, 16, 32, 0.7)',
                    bordercolor='rgba(167, 139, 250, 0.2)',
                    borderwidth=1,
                    thickness=14,
                    len=0.75
                ),
                hovertemplate='<b>%{y} %{x}</b><br>Mean Temp: <b>%{z:.1f}°C</b><extra></extra>'
            ))
            apply_dark_theme(fig_city_heat, height=370)
            fig_city_heat.update_layout(
                xaxis=dict(showgrid=False, tickfont=dict(color='#CBD5E1')),
                yaxis=dict(showgrid=False, tickfont=dict(color='#CBD5E1'))
            )
            st.plotly_chart(fig_city_heat, use_container_width=True)

        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("droplet", "#A78BFA", 14)} DAILY MEAN PRECIPITATION</div>
                    <div class="metric-value">{city_data['Rainfall'].mean():.2f} <span style="font-size: 1.1rem;">mm</span></div>
                    <div class="metric-subtitle">Decadal daily average</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("droplet", "#34D399", 14)} 10-YEAR CUMULATIVE</div>
                    <div class="metric-value" style="background: linear-gradient(135deg, #34D399, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{city_data['Rainfall'].sum():,.0f} <span style="font-size: 1.1rem;">mm</span></div>
                    <div class="metric-subtitle">Total recorded rainfall</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                heavy_rain_days = len(city_data[city_data['Rainfall'] > 35])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{get_svg_icon("cloud-rain", "#FBBF24", 14)} HEAVY MONSOON DAYS</div>
                    <div class="metric-value">{heavy_rain_days} <span style="font-size: 1.1rem;">days</span></div>
                    <div class="metric-subtitle">Daily precipitation &gt; 35 mm</div>
                </div>
                """, unsafe_allow_html=True)

            yearly_rain = city_data.groupby('Year')['Rainfall'].sum().reset_index()
            fig_rain_yr = go.Figure(data=[go.Bar(
                x=yearly_rain['Year'],
                y=yearly_rain['Rainfall'],
                marker=dict(
                    color=yearly_rain['Rainfall'],
                    colorscale='Teal',
                    line=dict(color='rgba(255,255,255,0.15)', width=1)
                ),
                hovertemplate='Year %{x}: %{y:,.0f} mm<extra></extra>'
            )])
            apply_dark_theme(fig_rain_yr, f"Annual Cumulative Precipitation Trajectory • {selected_city}", height=350)
            st.plotly_chart(fig_rain_yr, use_container_width=True)

        with tab3:
            comp_city = st.selectbox("Compare Station Against", [c for c in cities if c != selected_city], index=1 if len(cities) > 1 else 0)
            comp_data = df[df['City'] == comp_city].copy()

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_comp_temp = go.Figure(data=[
                    go.Bar(
                        name=selected_city,
                        x=['Mean Temperature (°C)'],
                        y=[city_data['Temperature'].mean()],
                        marker=dict(color='#A78BFA')
                    ),
                    go.Bar(
                        name=comp_city,
                        x=['Mean Temperature (°C)'],
                        y=[comp_data['Temperature'].mean()],
                        marker=dict(color='#F87171')
                    )
                ])
                apply_dark_theme(fig_comp_temp, f"Thermal Baseline: {selected_city} vs {comp_city}", height=340)
                fig_comp_temp.update_layout(barmode='group')
                st.plotly_chart(fig_comp_temp, use_container_width=True)

            with col_c2:
                fig_comp_rain = go.Figure(data=[
                    go.Bar(
                        name=selected_city,
                        x=['Mean Daily Rainfall (mm)'],
                        y=[city_data['Rainfall'].mean()],
                        marker=dict(color='#34D399')
                    ),
                    go.Bar(
                        name=comp_city,
                        x=['Mean Daily Rainfall (mm)'],
                        y=[comp_data['Rainfall'].mean()],
                        marker=dict(color='#C084FC')
                    )
                ])
                apply_dark_theme(fig_comp_rain, f"Precipitation Baseline: {selected_city} vs {comp_city}", height=340)
                fig_comp_rain.update_layout(barmode='group')
                st.plotly_chart(fig_comp_rain, use_container_width=True)

# ----------------- SCREEN 6: SCENARIO SIMULATOR -----------------
elif page == "Scenario Simulator":
    st.markdown(f'<p class="main-header">Atmospheric AI Scenario Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Interactive sensitivity testing and 2D Iso-Contour response surfaces using trained XGBoost ML models.</p>', unsafe_allow_html=True)
    st.markdown("---")

    if temp_model and rain_model and df_featured is not None:
        cities = sorted(df_featured['City'].unique())
        selected_city = st.selectbox("Select Station for Atmospheric Simulation", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)

        city_latest = df_featured[df_featured['City'] == selected_city].sort_values('Date').iloc[-1]

        st.markdown(f'<p class="section-title">{get_svg_icon("sliders", "#A78BFA", 18)} Adjust Atmospheric Input Parameters</p>', unsafe_allow_html=True)

        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            st.markdown("**Temperature Domain**")
            sim_temp = st.slider("Ambient Temperature (°C)", 5.0, 50.0, float(city_latest['Temperature']), 0.5)
            sim_temp_lag1 = st.slider("Prior Day Temp (Lag 1) (°C)", 5.0, 50.0, float(city_latest['Temp_Lag1']), 0.5)
            sim_temp_roll7 = st.slider("7-Day Mean Temp (°C)", 5.0, 50.0, float(city_latest['Temp_Roll7']), 0.5)

        with col_in2:
            st.markdown("**Precipitation Domain**")
            sim_rain = st.slider("Rainfall (mm)", 0.0, 150.0, float(city_latest['Rainfall']), 1.0)
            sim_rain_lag1 = st.slider("Prior Day Rain (Lag 1) (mm)", 0.0, 150.0, float(city_latest['Rain_Lag1']), 1.0)
            sim_rain_lag3 = st.slider("Prior 3-Day Rain (Lag 3) (mm)", 0.0, 150.0, float(city_latest['Rain_Lag3']), 1.0)

        with col_in3:
            st.markdown("**Dynamics & Seasonality**")
            sim_wind = st.slider("Wind Speed (km/h)", 0.0, 80.0, float(city_latest['WindSpeed']), 1.0)
            sim_month = st.slider("Calendar Month", 1, 12, int(city_latest['Month']))
            sim_doy = int(sim_month * 30.4)

        # Build feature vectors
        season_code = 0 if sim_month in [12, 1, 2] else (1 if sim_month in [3, 4, 5] else (2 if sim_month in [6, 7, 8] else 3))

        sim_features = {
            'Temperature': sim_temp,
            'Rainfall': sim_rain,
            'WindSpeed': sim_wind,
            'Temp_Lag1': sim_temp_lag1,
            'Temp_Lag3': float(city_latest['Temp_Lag3']),
            'Temp_Lag7': float(city_latest['Temp_Lag7']),
            'Temp_Roll3': float(city_latest['Temp_Roll3']),
            'Temp_Roll7': sim_temp_roll7,
            'Rain_Lag1': sim_rain_lag1,
            'Rain_Lag3': sim_rain_lag3,
            'Wind_Lag1': float(city_latest['Wind_Lag1']),
            'latitude': float(city_latest['latitude']),
            'longitude': float(city_latest['longitude']),
            'Month': sim_month,
            'DayOfYear': sim_doy,
            'DayOfYear_Sin': np.sin(2 * np.pi * sim_doy / 365.25),
            'DayOfYear_Cos': np.cos(2 * np.pi * sim_doy / 365.25),
            'Month_Sin': np.sin(2 * np.pi * sim_month / 12.0),
            'Month_Cos': np.cos(2 * np.pi * sim_month / 12.0),
            'Season_Encoded': season_code
        }

        X_sim_df = pd.DataFrame([sim_features])

        X_sim_temp = X_sim_df[temp_metrics['features']]
        X_sim_rain = X_sim_df[rain_metrics['features']]

        pred_sim_temp = float(temp_model.predict(X_sim_temp)[0])
        pred_sim_rain_code = int(rain_model.predict(X_sim_rain)[0])
        pred_sim_rain_label = str(rain_encoder.inverse_transform([pred_sim_rain_code])[0])

        base_pred_temp = float(temp_model.predict(pd.DataFrame([city_latest[temp_metrics['features']]]))[0])
        base_pred_rain_code = int(rain_model.predict(pd.DataFrame([city_latest[rain_metrics['features']]]))[0])
        base_pred_rain = str(rain_encoder.inverse_transform([base_pred_rain_code])[0])

        # 2D MACHINE LEARNING ISO-CONTOUR SENSITIVITY HEATMAP
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<p class="section-title">{get_svg_icon("cpu", "#A78BFA", 18)} 2D ML Iso-Contour Sensitivity Heatmap (Temperature vs Rainfall Space)</p>', unsafe_allow_html=True)

        temp_grid = np.linspace(10, 48, 25)
        rain_grid = np.linspace(0, 100, 25)
        T_mesh, R_mesh = np.meshgrid(temp_grid, rain_grid)

        grid_features = []
        for t_val, r_val in zip(T_mesh.ravel(), R_mesh.ravel()):
            f_copy = sim_features.copy()
            f_copy['Temperature'] = t_val
            f_copy['Rainfall'] = r_val
            grid_features.append(f_copy)

        df_grid = pd.DataFrame(grid_features)[temp_metrics['features']]
        Z_preds = temp_model.predict(df_grid).reshape(T_mesh.shape)

        fig_contour = go.Figure(data=go.Contour(
            z=Z_preds,
            x=temp_grid,
            y=rain_grid,
            colorscale='Plasma',
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(
                title="Predicted Temp (°C)",
                tickfont=dict(color="#CBD5E1", size=10),
                bgcolor='rgba(12, 16, 32, 0.7)',
                bordercolor='rgba(167, 139, 250, 0.2)',
                thickness=14,
                len=0.75
            ),
            hovertemplate='Input Temp: %{x:.1f}°C<br>Input Rain: %{y:.1f} mm<br><b>Predicted Next-Day Temp: %{z:.2f}°C</b><extra></extra>'
        ))

        # Add crosshair marker for current simulation point
        fig_contour.add_trace(go.Scatter(
            x=[sim_temp],
            y=[sim_rain],
            mode='markers',
            name='Current Operating Point',
            marker=dict(size=14, color='#34D399', symbol='cross', line=dict(width=2, color='white')),
            hovertemplate='Operating Point: Temp %{x:.1f}°C, Rain %{y:.1f} mm<extra></extra>'
        ))

        apply_dark_theme(fig_contour, f"AI Model Temperature Sensitivity Response Surface • {selected_city}", height=420)
        fig_contour.update_layout(
            xaxis_title="Input Ambient Temperature (°C)",
            yaxis_title="Input Precipitation (mm)"
        )
        st.plotly_chart(fig_contour, use_container_width=True)

        st.markdown("---")
        st.markdown(f'<p class="section-title">{get_svg_icon("bar-chart", "#A78BFA", 18)} Simulation Comparative Outcomes</p>', unsafe_allow_html=True)

        col_out1, col_out2 = st.columns(2)

        with col_out1:
            fig_temp = go.Figure(data=[
                go.Bar(
                    name='Baseline (Historical)',
                    x=['Forecasted Temp'],
                    y=[base_pred_temp],
                    marker=dict(color='#A78BFA', line=dict(color='white', width=1)),
                    hovertemplate='Baseline: %{y:.1f}°C<extra></extra>'
                ),
                go.Bar(
                    name='Simulated Scenario',
                    x=['Forecasted Temp'],
                    y=[pred_sim_temp],
                    marker=dict(color='#F87171', line=dict(color='white', width=1)),
                    hovertemplate='Simulated: %{y:.1f}°C<extra></extra>'
                )
            ])
            apply_dark_theme(fig_temp, "Next-Day Temperature Comparison (°C)", height=320)
            fig_temp.update_layout(barmode='group')
            st.plotly_chart(fig_temp, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 14px;">
                    <div class="metric-title" style="justify-content: center;">BASELINE</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #A78BFA;">{base_pred_temp:.1f}°C</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                diff = pred_sim_temp - base_pred_temp
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 14px; border-color: {'#F87171' if diff > 0 else '#34D399'};">
                    <div class="metric-title" style="justify-content: center;">SIMULATED OUTCOME</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {'#F87171' if diff > 0 else '#34D399'};">
                        {pred_sim_temp:.1f}°C ({diff:+.1f}°C)
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_out2:
            rain_categories = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
            rain_values = {cat: 1 if cat == base_pred_rain else 0 for cat in rain_categories}
            sim_rain_values = {cat: 1 if cat == pred_sim_rain_label else 0 for cat in rain_categories}

            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                name='Baseline',
                x=rain_categories,
                y=[rain_values[cat] for cat in rain_categories],
                marker=dict(color='#34D399'),
                hovertemplate='Baseline: %{x}<extra></extra>'
            ))
            fig_rain.add_trace(go.Bar(
                name='Simulated',
                x=rain_categories,
                y=[sim_rain_values[cat] for cat in rain_categories],
                marker=dict(color='#C084FC'),
                hovertemplate='Simulated: %{x}<extra></extra>'
            ))
            apply_dark_theme(fig_rain, "Precipitation Classification Shift", height=320)
            fig_rain.update_layout(barmode='group')
            st.plotly_chart(fig_rain, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 14px;">
                    <div class="metric-title" style="justify-content: center;">BASELINE</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #34D399;">{base_pred_rain}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 14px; border-color: rgba(192, 132, 252, 0.4);">
                    <div class="metric-title" style="justify-content: center;">SIMULATED PREDICTION</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #C084FC;">{pred_sim_rain_label}</div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- SCREEN 7: MODEL PERFORMANCE -----------------
elif page == "Model Performance":
    st.markdown(f'<p class="main-header">Machine Learning Engine Benchmarks</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Rigorous validation metrics, feature importance rankings, and residual distributions evaluated on 36,450 held-out test observations (2018–2019).</p>', unsafe_allow_html=True)
    st.markdown("---")

    if temp_metrics and rain_metrics:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Temperature Predictor (XGBoost Regressor)")
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #A78BFA;">
                <div class="metric-title">{get_svg_icon("cpu", "#A78BFA", 14)} MEAN ABSOLUTE ERROR (MAE)</div>
                <div class="metric-value">{temp_metrics['mae']:.3f}°C</div>
                <div class="metric-subtitle">Evaluated on unseen 2018–2019 test set (36,450 records)</div>
            </div>
            """, unsafe_allow_html=True)

            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-title">ROOT MEAN SQUARED ERROR</div>
                    <div class="metric-value" style="font-size: 1.6rem;">{temp_metrics['rmse']:.3f}°C</div>
                </div>
                """, unsafe_allow_html=True)
            with c_b:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-title">VARIANCE EXPLAINED (R²)</div>
                    <div class="metric-value" style="font-size: 1.6rem; color: #34D399;">{temp_metrics['r2']*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            if temp_model and hasattr(temp_model, 'feature_importances_'):
                st.markdown(f'<p class="section-title">{get_svg_icon("bar-chart", "#A78BFA", 18)} Feature Importance Breakdown</p>', unsafe_allow_html=True)
                feat_importances = pd.Series(temp_model.feature_importances_, index=temp_metrics['features']).sort_values(ascending=True)

                fig_feat = go.Figure(data=[go.Bar(
                    x=feat_importances.values,
                    y=feat_importances.index,
                    orientation='h',
                    marker=dict(
                        color=feat_importances.values,
                        colorscale='Plasma',
                        line=dict(color='rgba(255,255,255,0.1)', width=1)
                    ),
                    hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
                )])
                apply_dark_theme(fig_feat, "XGBoost Feature Gini Importance", height=380)
                st.plotly_chart(fig_feat, use_container_width=True)

        with col2:
            st.markdown("### Rainfall Classifier (XGBoost 4-Tier)")
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #34D399;">
                <div class="metric-title">{get_svg_icon("droplet", "#34D399", 14)} MULTI-CLASS ACCURACY</div>
                <div class="metric-value">{rain_metrics['accuracy']*100:.2f}%</div>
                <div class="metric-subtitle">Evaluated on unseen 2018–2019 test set (36,450 records)</div>
            </div>
            """, unsafe_allow_html=True)

            c_c, c_d = st.columns(2)
            with c_c:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-title">WEIGHTED F1-SCORE</div>
                    <div class="metric-value" style="font-size: 1.6rem; color: #34D399;">{rain_metrics['f1']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_d:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-title">CLASSIFIER TARGET CLASSES</div>
                    <div class="metric-value" style="font-size: 1.6rem;">4 Tiers</div>
                </div>
                """, unsafe_allow_html=True)

            if rain_model and hasattr(rain_model, 'feature_importances_'):
                st.markdown(f'<p class="section-title">{get_svg_icon("bar-chart", "#34D399", 18)} Rainfall Predictor Features</p>', unsafe_allow_html=True)
                rain_feat_importances = pd.Series(rain_model.feature_importances_, index=rain_metrics['features']).sort_values(ascending=True)

                fig_rain_feat = go.Figure(data=[go.Bar(
                    x=rain_feat_importances.values,
                    y=rain_feat_importances.index,
                    orientation='h',
                    marker=dict(
                        color=rain_feat_importances.values,
                        colorscale='Teal',
                        line=dict(color='rgba(255,255,255,0.1)', width=1)
                    ),
                    hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
                )])
                apply_dark_theme(fig_rain_feat, "XGBoost Rainfall Feature Importance", height=380)
                st.plotly_chart(fig_rain_feat, use_container_width=True)
