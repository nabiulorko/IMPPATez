#!/usr/bin/env python3


import warnings
warnings.filterwarnings("ignore")

import base64
import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import urllib.parse
import re
import os
import shutil
import time
import zipfile
import io
import subprocess
import tempfile
import platform
import json as _json
import select
import pty
import glob
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolAlign

try:
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    STEREO_ENUM = True
except ImportError:
    STEREO_ENUM = False

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import distance_matrix

try:
    from prody import parsePDB, calcCenter, writePDB, calcRMSD
    PRODY_AVAILABLE = True
except ImportError:
    PRODY_AVAILABLE = False

try:
    from meeko import MoleculePreparation
    try:
        from meeko import PDBQTWriterLegacy as _MeekoWriter
        MEEKO_LEGACY = True
    except ImportError:
        MEEKO_LEGACY = False
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False

try:
    from dimorphite_dl import protonate_smiles as _dimorphite_protonate
    DIMORPHITE_AVAILABLE = True
except ImportError:
    DIMORPHITE_AVAILABLE = False

BASE = "https://cb.imsc.res.in/imppat"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_DIR = "outputs"

METAL_RESNAMES = {
    "MG","ZN","CA","MN","FE","CU","CO","NI","CD","HG","NA","K","HO",
    "LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU",
}
METAL_CHARGES = {
    "MG":2.0,"ZN":2.0,"CA":2.0,"MN":2.0,"FE":3.0,
    "CU":2.0,"CO":2.0,"NI":2.0,"CD":2.0,"HG":2.0,"HO":3.0,
    "LA":3.0,"CE":3.0,"PR":3.0,"ND":3.0,"PM":3.0,"SM":3.0,
    "EU":3.0,"GD":3.0,"TB":3.0,"DY":3.0,"ER":3.0,"TM":3.0,
    "YB":3.0,"LU":3.0,"NA":1.0,"K":1.0,
}
_NO_REINJECT = {"HO","LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU"}
EXCLUDE_IONS = set(
    "HOH,WAT,DOD,SOL,NA,CL,K,CA,MG,ZN,MN,FE,CU,CO,NI,CD,HG,HO,"
    "LA,CE,PR,ND,PM,SM,EU,GD,TB,DY,ER,TB,YB,LU".split(",")
)
GLYCAN_NAMES = {"NAG","BMA","MAN","FUC","GAL","GLC","SIA","NGA","FUL","GLA","BGC","A2G","LAT","MAL"}
COFACTOR_NAMES = {"ATP","ADP","AMP","GTP","GDP","GMP","NAD","NAP","NDP","FAD","FMN","HEM","HEC","HEA",
                  "GOL","PEG","EDO","MPD","PGE","PG4","SO4","PO4","SUL","PHO","IHP","TTP","CTP","UTP",
                  "COA","SAM","SAH","EPE","MES","TRS","ACT","ACY"}
HEME_RESNAMES = {"HEM","HEC","HEA","HEB","HDD","HDM"}
_MIN_LIG_ATOMS = 4
_BACKBONE = {"N","CA","C","O"}

st.set_page_config(
    page_title="IMPPATez — Natural Product Docking",
    page_icon="🍀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=IBM+Plex+Mono:wght@500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; font-size: 0.82rem; }
    p, span, div, label, li, a, td, th, input, textarea, button { font-family: 'Poppins', sans-serif; }
    :root {
        --bg:#FFFFFF; --bg-subtle:#F6F8FA; --border:#D0D7DE;
        --green:#2e7d32; --green-dark:#1b5e20;
        --success:#2e7d32; --warn:#9A6700; --danger:#c62828;
        --text:#24292F; --text-muted:#57606A;
    }

    /* Header */
    .main-title { font-size: 2.6rem; font-weight: 800; margin-bottom: 0.15rem; line-height: 1.2; }
    .main-title .logo-emoji { font-size: 1.35em; vertical-align: middle; margin-right: 0.05em; display: inline-block; }
    .imppat { color: #1a3d20; }
    .ez { color: var(--green); }
    .subtitle { color: #555; font-size: 0.78rem; margin-bottom: 0.9rem; }
    }

    /* Step cards */
    .step-card {
        background: #f8f9fa; border-left: 4px solid var(--green);
        border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.9rem;
    }
    .step-title {
        font-weight: 800; color: var(--green); font-size: 1.55rem;
        text-transform: uppercase; letter-spacing: 0.6px; margin: 0;
        font-family: 'Poppins', sans-serif;
    }
    .step-heading { color: #555; font-size: 0.78rem; margin-top: 0.2rem; }

    /* Pills / badges */
    .result-pill {
        display: inline-block; background: #e3f2fd; border: 1px solid #64b5f6;
        color: #0d47a1; border-radius: 20px; padding: 2px 12px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; margin: 2px;
    }
    .success-pill {
        display: inline-block; background: #e8f5e9; border: 1px solid var(--green);
        color: var(--green); border-radius: 20px; padding: 4px 14px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
    }
    .warn-pill {
        display: inline-block; background: #fff8e1; border: 1px solid var(--warn);
        color: var(--warn); border-radius: 20px; padding: 4px 14px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
    }

    /* Stat / score displays */
    .stat-card { background: white; border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid #e0e0e0; }
    .stat-number { font-size: 1.8rem; font-weight: 800; color: var(--green); }
    .score-best { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; color: var(--green); font-weight: 700; }
    .score-unit { font-size: 0.95rem; color: var(--text-muted); }

    .log-box {
        background: var(--bg-subtle); border: 1px solid var(--border); border-radius: 6px;
        padding: 12px 16px; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
        max-height: 220px; overflow-y: auto; white-space: pre-wrap;
    }

    /* Buttons: flat, precise, with a soft glow */
    @keyframes btn-glow {
        0%, 100% { box-shadow: 0 0 4px rgba(46,125,50,0.55), 0 0 10px rgba(46,125,50,0.30); }
        50%      { box-shadow: 0 0 12px rgba(46,125,50,0.85), 0 0 24px rgba(46,125,50,0.55); }
    }
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        background: var(--green); color: white; border: none;
        font-weight: 600; border-radius: 6px; padding: 0.5rem 1.25rem;
        transition: background 0.15s ease, box-shadow 0.15s ease;
        animation: btn-glow 2.2s ease-in-out infinite;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
        background: var(--green-dark);
        box-shadow: 0 0 16px rgba(46,125,50,0.95), 0 0 32px rgba(46,125,50,0.65);
    }
    .stButton > button[kind="secondary"], .stFormSubmitButton > button[kind="secondary"] {
        background: #f1f3f4; color: var(--text); border: 1px solid var(--border);
        animation: btn-glow-secondary 2.2s ease-in-out infinite;
    }
    @keyframes btn-glow-secondary {
        0%, 100% { box-shadow: 0 0 3px rgba(46,125,50,0.30), 0 0 8px rgba(46,125,50,0.15); }
        50%      { box-shadow: 0 0 10px rgba(46,125,50,0.55), 0 0 18px rgba(46,125,50,0.30); }
    }
    .stButton > button[kind="secondary"]:hover, .stFormSubmitButton > button[kind="secondary"]:hover { background: #e8eaed; }

    .engine-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 700;
    }
    hr { margin: 1rem 0; border: none; border-top: 1px solid #e0e0e0; }
    .rmsd-excellent { color: var(--green); font-weight: 700; }
    .rmsd-good { color: var(--warn); font-weight: 700; }
    .rmsd-poor { color: var(--danger); font-weight: 700; }

    .current-receptor-card {
        background: #f8f9fa; border-left: 4px solid var(--green);
        border-radius: 6px; padding: 12px 16px; margin: 14px 0;
    }
    .current-receptor-label {
        font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 4px; font-weight: 700;
    }
    .current-receptor-path { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; color: #1a3d20; overflow-wrap: anywhere; }

    .ligand-detect-card {
        display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr);
        gap: 12px; background: #f8f9fa; border-left: 4px solid var(--green);
        border-radius: 6px; padding: 14px; margin: 12px 0;
    }
    .ligand-detect-item { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; }
    .ligand-detect-label {
        color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 1px; font-weight: 700; margin-bottom: 5px;
    }
    .ligand-detect-value { color: #163b22; font-size: 1rem; font-weight: 700; overflow-wrap: anywhere; }
    @media (max-width: 760px) { .ligand-detect-card { grid-template-columns: 1fr; } }

    .rmsd-card { background: #f8f9fa; border-radius: 6px; padding: 16px; margin: 12px 0; border-left: 4px solid var(--green); }
    .rmsd-value { font-size: 1.7rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

    /* Pipeline stepper */
    .st-key-pipeline_stepper_wrap { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 16px 16px 12px 16px; margin-bottom: 1.4rem; }
    .st-key-pipeline_stepper_wrap [data-testid="column"] { display: flex; flex-direction: column; align-items: center; }
    .st-key-pipeline_stepper_wrap div[data-testid="stButton"] button {
        border-radius: 6px !important; font-weight: 700 !important; font-size: 0.8rem !important; padding: 8px 6px !important;
    }
    .st-key-pipeline_stepper_wrap div[data-testid="stButton"] button[kind="secondary"] {
        background: #f6f8fa !important; color: #5b6670 !important; border: 1px solid #e0e0e0 !important;
    }
    .st-key-pipeline_stepper_wrap div[data-testid="stButton"] button[kind="primary"] {
        background: var(--green) !important; border-color: var(--green) !important; color: #fff !important;
    }
    .st-key-pipeline_stepper_wrap div[data-testid="stButton"] button[kind="primary"]:hover { background: var(--green-dark) !important; }

    .prereq-banner {
        display: flex; align-items: center; gap: 10px;
        background: #fff8e1; border-left: 4px solid var(--warn);
        border-radius: 6px; padding: 10px 14px; margin-bottom: 1rem;
        font-size: 0.86rem; color: #6b5200;
    }

    /* ADMET panel */
    .admet-source-tag {
        display: inline-flex; align-items: center; gap: 6px;
        border-radius: 20px; padding: 3px 12px; font-size: 0.72rem;
        font-weight: 700; margin-bottom: 8px; font-family: 'IBM Plex Mono', monospace;
    }
    .admet-strip {
        display: flex; flex-wrap: wrap; gap: 0; background: #f8f9fa;
        border-left: 4px solid var(--green); border-radius: 6px;
        padding: 12px 4px; margin: 8px 0;
    }
    .admet-stat {
        flex: 1 1 100px; min-width: 90px; padding: 4px 14px;
        border-right: 1px solid #e2e5e9;
    }
    .admet-stat:last-child { border-right: none; }
    .admet-stat-label {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.8px;
        color: var(--text-muted); font-weight: 700; margin-bottom: 2px;
    }
    .admet-stat-value {
        font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700;
    }
    .admet-stat-unit { font-size: 0.62rem; font-weight: 400; color: #8a8f98; margin-left: 3px; }
    .admet-chip {
        display: inline-flex; align-items: center; gap: 4px;
        border-radius: 4px; padding: 3px 10px; font-size: 0.72rem; font-weight: 700;
        font-family: 'IBM Plex Mono', monospace; margin: 2px 4px 2px 0; border: 1px solid transparent;
    }
    .admet-verdict-card {
        background: #ffffff; border: 1px solid #e0e0e0; border-radius: 6px;
        padding: 10px 14px; text-align: left; height: 100%;
    }
    .admet-verdict-label {
        font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase;
        letter-spacing: 0.8px; font-weight: 700; margin-bottom: 4px;
    }
    .admet-verdict-value { font-size: 1rem; font-weight: 700; }

    /* Footer */
    .footer-bar {
        background: linear-gradient(90deg, #0d1b2a 0%, #1a3a5c 100%);
        border-radius: 10px; padding: 1rem 1.5rem; margin-top: 1.5rem;
        display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.5rem;
    }
    .footer-brand { font-size: 1.1rem; font-weight: 800; color: #fff; letter-spacing: 0.5px; }
    .footer-brand span { color: #66bb6a; }
    .footer-meta { font-size: 0.76rem; color: #a0b8cc; text-align: right; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


st.markdown("""
<div class="main-title"><span class="logo-emoji">🍀</span><span class="imppat">IMPPAT</span><span class="ez">ez</span></div>
<div class="subtitle">Integrated Platform for Automated IMPPAT Phytochemicals Extraction & Molecular Docking</div>

""", unsafe_allow_html=True)


def timestamp_tag():
    return datetime.now().strftime("%Y%m%d-%H%M")

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

@st.cache_resource
def get_session():
    return create_robust_session()

def safe_name(s):
    s = (s or "").strip() or "mol"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s

def receptor_session_tag(receptor_path):
    if receptor_path:
        stem = Path(str(receptor_path)).stem.strip()
        stem = re.sub(r"^receptor_", "", stem, flags=re.IGNORECASE)
        if re.fullmatch(r"[0-9A-Za-z]{4}", stem):
            return stem.upper()
        if stem:
            return safe_name(stem)
    return "receptor"

def default_docking_session_name(receptor_path=None):
    return f"imppatez_{receptor_session_tag(receptor_path)}_{timestamp_tag()}"

def assert_file_ok(path, min_bytes=100, msg=""):
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise ValueError(msg or f"File missing/too small: {path}")

def trigger_browser_download(file_path):
    """Auto-trigger a browser download of file_path with no extra click required."""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    fname = os.path.basename(file_path)
    components.html(
        f"""
        <html><body>
        <a id="auto_dl_link" href="data:application/zip;base64,{b64}" download="{fname}"></a>
        <script>
            document.getElementById('auto_dl_link').click();
        </script>
        </body></html>
        """,
        height=0, width=0,
    )

def unique_workdir(base_path):
    """Return a non-existing path by adding a numeric suffix when needed."""
    base = Path(base_path)
    if not base.exists():
        return base
    for i in range(2, 1000):
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
    return base.with_name(f"{base.name}_{timestamp_tag()}")

def unique_filepath(base_path):
    """Return a non-existing FILE path by adding a clean numeric suffix
    before the extension when needed (e.g. receptor.pdb, receptor_2.pdb)
    instead of cluttering names with timestamps."""
    base = Path(base_path)
    if not base.exists():
        return str(base)
    stem, suffix = base.stem, base.suffix
    for i in range(2, 1000):
        candidate = base.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return str(candidate)
    return str(base.with_name(f"{stem}_{timestamp_tag()}{suffix}"))


PIPELINE_STEPS = [
    {"label": "Phytochemical Extraction", "sub": "1 · Extraction"},
    {"label": "Docking Engine", "sub": "2 · Setup"},
    {"label": "Receptor Preparation",       "sub": "3 · Prep"},
    {"label": "Ligand Preparation",        "sub": "4 · Prep"},
    {"label": "Redocking & Validation",      "sub": "5 · Validate"},
    {"label": "Batch Docking",  "sub": "6 · Run & Results"},
]

def pipeline_step_status():
    """Return a list of booleans marking each of the 6 pipeline steps as complete."""
    ss = st.session_state
    return [
        bool(ss.get("df") is not None or ss.get("csv_path")),
        bool(ss.get("dock_bin_path")),
        bool(ss.get("receptor_result") or ss.get("receptor_pdb_path")),
        bool(ss.get("selected_smiles_list")),
        bool(ss.get("redock_result")),
        bool(ss.get("dock_records")),
    ]

def render_pipeline_stepper():

    if "active_step" not in st.session_state:
        st.session_state.active_step = 1

    done_flags = pipeline_step_status()
    active = st.session_state.active_step

    with st.container(key="pipeline_stepper_wrap"):
        cols = st.columns(len(PIPELINE_STEPS))
        for i, (col, step, done) in enumerate(zip(cols, PIPELINE_STEPS, done_flags), start=1):
            with col:
                is_active = (active == i)
                icon = "✅" if done else f"{i}."
                if st.button(
                    f"{icon}  {step['label']}",
                    key=f"stepper_btn_{i}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    help=step["sub"],
                ):
                    st.session_state.active_step = i
                    st.rerun()

def render_prereq_banner(step_index, message):

    done_flags = pipeline_step_status()
    if not done_flags[step_index]:
        st.markdown(
            f'<div class="prereq-banner">⚠️ {message}</div>',
            unsafe_allow_html=True,
        )


def extract_ligand_coords_from_pdb(pdb_file, ligand_resname=None, ligand_chain=None, ligand_resid=None):
    coords = []
    atoms = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                if ligand_resname:
                    resname = line[17:20].strip()
                    if resname != ligand_resname:
                        continue
                if ligand_chain is not None and line[21].strip() != str(ligand_chain).strip():
                    continue
                if ligand_resid is not None:
                    try:
                        if int(line[22:26]) != int(ligand_resid):
                            continue
                    except Exception:
                        continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    atom_name = line[12:16].strip()
                    atoms.append(atom_name)
                except:
                    continue
    return np.array(coords), atoms

def calculate_rmsd(coords1, coords2):
    if len(coords1) != len(coords2):
        min_len = min(len(coords1), len(coords2))
        coords1 = coords1[:min_len]
        coords2 = coords2[:min_len]
    if len(coords1) == 0:
        return None
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def calculate_rmsd_with_alignment(coords1, coords2):
    if len(coords1) != len(coords2):
        min_len = min(len(coords1), len(coords2))
        coords1 = coords1[:min_len]
        coords2 = coords2[:min_len]
    if len(coords1) == 0:
        return None
    
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    
    H = np.dot(coords1_centered.T, coords2_centered)
    try:
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
        if d < 0:
            Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    except np.linalg.LinAlgError:
        R = np.eye(3)
    
    coords2_aligned = np.dot(coords2_centered, R)
    diff = coords1_centered - coords2_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def _heavy_mol(mol):
    if mol is None:
        return None
    try:
        return Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        return Chem.RemoveHs(mol)

def _assign_template_bonds(template, mol):
    if mol is None:
        return None
    if template is None:
        return _heavy_mol(mol)
    try:
        templ = _heavy_mol(template)
        target = _heavy_mol(mol)
        assigned = AllChem.AssignBondOrdersFromTemplate(templ, target)
        Chem.SanitizeMol(assigned)
        return assigned
    except Exception:
        return _heavy_mol(mol)

def _first_sdf_mol(sdf_path):
    supp = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    return next((m for m in supp if m is not None), None)

def _sdf_mols(sdf_path):
    supp = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    return [m for m in supp if m is not None]

def _pose_affinity(mol):
    if mol is None:
        return None
    for prop in ("minimizedAffinity", "affinity", "docking_score", "ENERGY"):
        if mol.HasProp(prop):
            try:
                return float(re.findall(r"[-+]?\d*\.?\d+", mol.GetProp(prop))[0])
            except Exception:
                pass
    return None

def _direct_rmsd_for_match(ref_mol, docked_mol, ref_match, docked_match):
    ref_conf = ref_mol.GetConformer()
    docked_conf = docked_mol.GetConformer()
    sq = []
    for ref_idx, docked_idx in zip(ref_match, docked_match):
        rp = ref_conf.GetAtomPosition(int(ref_idx))
        dp = docked_conf.GetAtomPosition(int(docked_idx))
        sq.append((rp.x - dp.x) ** 2 + (rp.y - dp.y) ** 2 + (rp.z - dp.z) ** 2)
    if not sq:
        return None
    return float(np.sqrt(np.mean(sq)))

def _rmsd_against_reference(ref_mol, docked_mol, query=None):
    if ref_mol is None or docked_mol is None:
        return None
    
    if query is not None:
        ref_matches = ref_mol.GetSubstructMatches(query, uniquify=False, maxMatches=512)
        docked_matches = docked_mol.GetSubstructMatches(query, uniquify=False, maxMatches=512)
        if ref_matches and docked_matches:
            best = float('inf')
            for ref_match in ref_matches:
                for docked_match in docked_matches:
                    val = _direct_rmsd_for_match(ref_mol, docked_mol, ref_match, docked_match)
                    if val is not None and val < best:
                        best = val
            if best < float('inf'):
                return best
    
    ref_heavy = _heavy_mol(ref_mol)
    docked_heavy = _heavy_mol(docked_mol)
    
    if ref_heavy is None or docked_heavy is None:
        return None
    
    ref_conf = ref_heavy.GetConformer()
    docked_conf = docked_heavy.GetConformer()
    
    ref_coords = []
    for atom in ref_heavy.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = ref_conf.GetAtomPosition(atom.GetIdx())
        ref_coords.append([pos.x, pos.y, pos.z])
    
    docked_coords = []
    for atom in docked_heavy.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = docked_conf.GetAtomPosition(atom.GetIdx())
        docked_coords.append([pos.x, pos.y, pos.z])
    
    ref_coords = np.array(ref_coords)
    docked_coords = np.array(docked_coords)
    
    if len(ref_coords) != len(docked_coords) or len(ref_coords) == 0:
        return None
    
    return calculate_rmsd_with_alignment(ref_coords, docked_coords)

def calculate_rmsd_between_ligands(ref_ligand_path, docked_ligand_path, ligand_resname, ligand_smiles=None, return_details=False):
    try:
        template = Chem.MolFromSmiles(ligand_smiles) if ligand_smiles else None
        
        ref_mol = Chem.MolFromPDBFile(str(ref_ligand_path), sanitize=False, removeHs=False, proximityBonding=True)
        docked_mols = _sdf_mols(docked_ligand_path)

        if ref_mol is None:
            return None
        
        if not docked_mols:
            return None

        ref_mol = _assign_template_bonds(template, ref_mol)
        query = _heavy_mol(template) if template is not None else None

        pose_details = []
        for pose_idx, mol in enumerate(docked_mols, start=1):
            try:
                docked_mol = _assign_template_bonds(template, mol)
                val = _rmsd_against_reference(ref_mol, docked_mol, query)
                if val is not None:
                    pose_details.append({
                        "pose": pose_idx,
                        "rmsd": val,
                        "affinity": _pose_affinity(mol),
                    })
            except Exception as e:
                if hasattr(st, "debug"):
                    st.debug(f"RMSD error for pose {pose_idx}: {e}")
                continue

        if not pose_details:
            return None
        
        best_pose = min(pose_details, key=lambda d: d["rmsd"])
        top_pose = pose_details[0]
        
        if return_details:
            return {
                "top_pose_rmsd": top_pose["rmsd"],
                "best_pose_rmsd": best_pose["rmsd"],
                "best_pose": best_pose["pose"],
                "pose_rmsds": pose_details,
            }
        return top_pose["rmsd"]
        
    except Exception as e:
        if hasattr(st, "debug"):
            st.debug(f"RMSD calculation failed: {e}")
        return None

def calculate_rmsd_prody(ref_pdb, docked_pdb, ligand_resname):
    if not PRODY_AVAILABLE:
        return None
    try:
        ref = parsePDB(ref_pdb)
        docked = parsePDB(docked_pdb)
        ref_lig = ref.select(f"resname {ligand_resname}")
        docked_lig = docked.select(f"resname {ligand_resname}")
        if ref_lig is None or docked_lig is None:
            return None
        rmsd = calcRMSD(ref_lig, docked_lig)
        return rmsd
    except Exception as e:
        if hasattr(st, "debug"):
            st.debug(f"ProDy RMSD calculation failed: {e}")
        return None

def get_rmsd_validation_message(rmsd):
    if rmsd is None:
        return "❓ Could not calculate RMSD", "unknown"
    elif rmsd < 1.5:
        return f"✅ EXCELLENT! RMSD = {rmsd:.3f} Å (Docking protocol is highly reliable)", "excellent"
    elif rmsd < 2.0:
        return f"✅ GOOD! RMSD = {rmsd:.3f} Å (Docking protocol is reliable)", "good"
    elif rmsd < 3.0:
        return f"⚠️ Needs Optimization = {rmsd:.3f} Å (Protocol may need optimization)", "acceptable"
    else:
        return f"❌ POOR - RMSD = {rmsd:.3f} Å (Docking protocol needs significant improvement)", "poor"

def quick_rmsd_check_debug(smiles_file, sdf_file, smiles_string=None):
    try:
        if smiles_string:
            template = Chem.MolFromSmiles(smiles_string)
        else:
            template = None
        
        ref_mol = Chem.MolFromMol2File(smiles_file, sanitize=False) if smiles_file.endswith('.mol2') else Chem.MolFromPDBFile(smiles_file, sanitize=False)
        docked_mol = _first_sdf_mol(sdf_file)
        if ref_mol is None or docked_mol is None:
            return None
        ref_mol = _assign_template_bonds(template, ref_mol)
        query = _heavy_mol(template) if template is not None else None
        return _rmsd_against_reference(ref_mol, docked_mol, query)
    except Exception:
        return None


def ligand_box_from_coords(coords, padding=8.0, min_size=12.0, max_size=24.0):
    """Create a compact redocking box around the crystallographic ligand."""
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return None
    center = coords.mean(axis=0)
    span = coords.max(axis=0) - coords.min(axis=0)
    size = np.clip(span + float(padding), float(min_size), float(max_size))
    return tuple(float(x) for x in (*center, *size))

def write_vina_config(config_path, cx, cy, cz, sx, sy, sz):
    with open(config_path, "w") as f:
        f.write(
            f"center_x = {cx:.4f}\n"
            f"center_y = {cy:.4f}\n"
            f"center_z = {cz:.4f}\n"
            f"size_x = {float(sx):.2f}\n"
            f"size_y = {float(sy):.2f}\n"
            f"size_z = {float(sz):.2f}\n"
        )

def write_box_pdb(filename, cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx/2, sy/2, sz/2
    corners = [(cx+dx, cy+dy, cz+dz) for dx in (-hx,hx) for dy in (-hy,hy) for dz in (-hz,hz)]
    with open(filename, "w") as f:
        for i, (x,y,z) in enumerate(corners, 1):
            f.write(f"HETATM{i:5d}  C   BOX A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("CONECT    1    2    3    5\nCONECT    2    1    4    6\nCONECT    3    1    4    7\nCONECT    4    2    3    8\nCONECT    5    1    6    7\nCONECT    6    2    5    8\nCONECT    7    3    5    8\nCONECT    8    4    6    7\n")

def render_redocking_pose_viewer(original_ligand_pdb, docked_sdf, height=520):
    if not original_ligand_pdb or not docked_sdf:
        return
    if not os.path.exists(original_ligand_pdb) or not os.path.exists(docked_sdf):
        return
    try:
        with open(original_ligand_pdb, "r") as f:
            original_pdb = f.read()
        with open(docked_sdf, "r") as f:
            docked_sdf_text = f.read()
    except Exception as e:
        st.warning(f"Could not load pose viewer files: {e}")
        return

    viewer_id = f"redock_viewer_{int(time.time() * 1000)}"
    html = f"""
    <div style="border:1px solid #d0d7de;border-radius:8px;overflow:hidden;background:#ffffff;">
      <div id="{viewer_id}" style="width:100%;height:{height}px;"></div>
      <div style="display:flex;gap:18px;align-items:center;padding:8px 12px;font-family:IBM Plex Sans,Arial,sans-serif;font-size:13px;border-top:1px solid #d0d7de;background:#f6f8fa;">
        <span><span style="display:inline-block;width:12px;height:12px;background:#d62728;border-radius:2px;margin-right:6px;"></span>Co-crystal pose</span>
        <span><span style="display:inline-block;width:12px;height:12px;background:#1f4fd6;border-radius:2px;margin-right:6px;"></span>Redocked pose</span>
      </div>
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      var originalPdb = {_json.dumps(original_pdb)};
      var dockedSdf = {_json.dumps(docked_sdf_text)};
      var viewer = $3Dmol.createViewer("{viewer_id}", {{backgroundColor: "white"}});
      viewer.addModel(originalPdb, "pdb");
      viewer.setStyle({{model: 0}}, {{stick: {{color: "red", radius: 0.18}}, sphere: {{color: "red", scale: 0.22}}}});
      viewer.addModel(dockedSdf, "sdf");
      viewer.setStyle({{model: 1}}, {{stick: {{color: "blue", radius: 0.18}}, sphere: {{color: "blue", scale: 0.22}}}});
      viewer.zoomTo();
      viewer.render();
    </script>
    """
    components.html(html, height=height + 48)


def redock_cocrystal_ligand_from_smiles(rec_result, cocrystal_smiles, cocrystal_resname, 
                                         engine, dock_bin, dock_params, workdir, source_pdb=None,
                                         cocrystal_chain=None, cocrystal_resid=None):
    try:
        receptor_pdb = source_pdb or rec_result.get('raw_pdb') or rec_result.get('receptor_pdb')
        original_coords, original_atoms = extract_ligand_coords_from_pdb(
            receptor_pdb, cocrystal_resname, cocrystal_chain, cocrystal_resid
        )
        
        if len(original_coords) == 0:
            loc = f" chain {cocrystal_chain} resid {cocrystal_resid}" if cocrystal_chain or cocrystal_resid else ""
            st.error(f"Could not extract co-crystal ligand '{cocrystal_resname}'{loc} from original PDB file: {receptor_pdb}")
            return None
                
        original_ligand_pdb = str(workdir / f"original_{cocrystal_resname}.pdb")
        with open(receptor_pdb, 'r') as f_in, open(original_ligand_pdb, 'w') as f_out:
            for line in f_in:
                if line.startswith(('ATOM', 'HETATM')):
                    resname = line[17:20].strip()
                    chain_ok = cocrystal_chain is None or line[21].strip() == str(cocrystal_chain).strip()
                    try:
                        resid_ok = cocrystal_resid is None or int(line[22:26]) == int(cocrystal_resid)
                    except Exception:
                        resid_ok = False
                    if resname == cocrystal_resname and chain_ok and resid_ok:
                        f_out.write(line)
            f_out.write("END\n")
        
        lig_name = f"cocrystal_{cocrystal_resname}"
        lig_prep = prepare_ligand_vina_batch(cocrystal_smiles, lig_name, 7.4, workdir)
        
        if not lig_prep['success']:
            st.error("Failed to prepare co-crystal ligand from SMILES")
            return None
        
        lig_input = lig_prep["sdf"] if engine == "UNIDOCK" else lig_prep["pdbqt"]
        out_prefix = str(workdir / f"redock_{cocrystal_resname}")
        config_file = rec_result["config_file"]
        
        redock_box = None
        if dock_params.get("use_crystal_box", True):
            redock_box = ligand_box_from_coords(
                original_coords,
                padding=dock_params.get("redock_box_padding", 8.0),
                min_size=dock_params.get("redock_min_box_size", 12.0),
                max_size=dock_params.get("redock_max_box_size", 24.0),
            )
            if redock_box:
                cx, cy, cz, sx, sy, sz = redock_box
                config_file = str(workdir / f"redock_{cocrystal_resname}.box.txt")
                box_pdb = str(workdir / f"redock_{cocrystal_resname}.box.pdb")
                write_vina_config(config_file, cx, cy, cz, sx, sy, sz)
                write_box_pdb(box_pdb, cx, cy, cz, sx, sy, sz)
        
        docked_pdbqt, docked_sdf, dock_log = dock_one_ligand(
            engine, dock_bin,
            rec_result["receptor_pdbqt"],
            lig_input, config_file,
            out_prefix, dock_params
        )
        
        if not docked_sdf or not os.path.exists(docked_sdf):
            st.error("Docking failed - no output file")
            return None
        
        rmsd_details = calculate_rmsd_between_ligands(
            original_ligand_pdb, docked_sdf, cocrystal_resname, cocrystal_smiles, return_details=True
        )
        
        if isinstance(rmsd_details, dict):
            rmsd = rmsd_details.get("top_pose_rmsd")
        else:
            rmsd = rmsd_details

        pose_scores = parse_all_poses(docked_pdbqt, docked_sdf, engine)
        pose_affinities = {
            p.get("pose"): p.get("affinity")
            for p in pose_scores
            if p.get("pose") is not None
        }
        binding_affinity = pose_affinities.get(1)
        if binding_affinity is None and pose_scores:
            binding_affinity = pose_scores[0].get("affinity")

        if isinstance(rmsd_details, dict):
            for row in rmsd_details.get("pose_rmsds") or []:
                if row.get("affinity") is None:
                    row["affinity"] = pose_affinities.get(row.get("pose"))
            rmsd_details["binding_affinity"] = binding_affinity
        
        rmsd_prody = None
        if PRODY_AVAILABLE:
            rmsd_prody = calculate_rmsd_prody(original_ligand_pdb, docked_sdf, cocrystal_resname)
        
        try:
            template = Chem.MolFromSmiles(cocrystal_smiles)
            n_heavy_docked = template.GetNumHeavyAtoms() if template else 0
        except:
            n_heavy_docked = 0
        
        return {
            'success': True,
            'rmsd': rmsd,
            'rmsd_prody': rmsd_prody,
            'docked_sdf': docked_sdf,
            'docked_pdbqt': docked_pdbqt,
            'original_ligand_pdb': original_ligand_pdb,
            'original_smiles': cocrystal_smiles,
            'original_resname': cocrystal_resname,
            'ligand_name': lig_name,
            'log': dock_log,
            'binding_affinity': binding_affinity,
            'n_atoms_original': len(original_coords),
            'n_atoms_docked': n_heavy_docked,
            'rmsd_method': 'heavy_atom_symmetry_aware_top_pose',
            'rmsd_details': rmsd_details if isinstance(rmsd_details, dict) else None,
            'redock_box': redock_box,
            'redock_config_file': config_file,
        }
    except Exception as e:
        st.error(f"Redocking failed: {e}")
        return None


def search_rcsb_pdb(query, top_n=20):
    try:
        if re.match(r'^[0-9a-zA-Z]{4}$', query.upper()):
            url = f"https://data.rcsb.org/rest/v1/core/entry/{query.upper()}"
            response = get_session().get(url, timeout=15)
            if response.status_code == 200:
                entry = response.json()
                return [{
                    "pdb_id": query.upper(),
                    "title": entry.get("struct", {}).get("title", ""),
                    "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                    "method": entry.get("exptl", [{}])[0].get("method", ""),
                    "organism": entry.get("rcsb_entity_source_organism", [{}])[0].get("common_name", ""),
                    "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date", "")
                }]
        
        payload = {
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": top_n}, "results_verbosity": "compact", "sort": [{"sort_by": "score", "direction": "desc"}]},
        }
        r = get_session().post("https://search.rcsb.org/rcsbsearch/v2/query", json=payload, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, dict):
            return []
        result_set = data.get("result_set", [])
        if not result_set:
            return []
        results = []
        for hit in result_set:
            try:
                pdb_id = hit.get("identifier", "") if isinstance(hit, dict) else str(hit) if hit else ""
                if not pdb_id:
                    continue
                detail_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                try:
                    r2 = get_session().get(detail_url, timeout=10)
                    if r2.status_code == 200:
                        entry = r2.json()
                        results.append({
                            "pdb_id": pdb_id,
                            "title": entry.get("struct", {}).get("title", "")[:120],
                            "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0] or "N/A",
                            "method": entry.get("exptl", [{}])[0].get("method", "") or "N/A",
                            "organism": entry.get("rcsb_entity_source_organism", [{}])[0].get("common_name", "") or "N/A",
                            "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date", "")
                        })
                    else:
                        results.append({"pdb_id": pdb_id, "title": "", "resolution": "N/A", "method": "N/A", "organism": "N/A", "deposition_date": ""})
                except Exception:
                    results.append({"pdb_id": pdb_id, "title": "", "resolution": "N/A", "method": "N/A", "organism": "N/A", "deposition_date": ""})
            except Exception:
                continue
        return results
    except Exception as e:
        st.error(f"RCSB search error: {str(e)}")
        return []

def download_pdb_direct(pdb_id, output_path):
    try:
        pdb_id = pdb_id.upper().strip()
        urls = [
            f"https://files.rcsb.org/download/{pdb_id}.pdb",
            f"https://models.rcsb.org/{pdb_id}.pdb",
            f"https://www.rcsb.org/pdb/files/{pdb_id}.pdb"
        ]
        for url in urls:
            try:
                response = get_session().get(url, timeout=30)
                if response.status_code == 200:
                    content = response.text
                    if content.strip().startswith(("HEADER", "ATOM", "HETATM", "CRYST1", "REMARK")):
                        with open(output_path, "w") as f:
                            f.write(content)
                        if os.path.getsize(output_path) > 1000:
                            return True
            except Exception:
                continue
        return False
    except Exception as e:
        st.error(f"Download error for {pdb_id}: {str(e)}")
        return False

def get_pdb_info(pdb_id):
    try:
        pdb_id = pdb_id.upper().strip()
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        response = get_session().get(url, timeout=10)
        if response.status_code == 200:
            entry = response.json()
            return {
                "title": entry.get("struct", {}).get("title", ""),
                "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                "method": entry.get("exptl", [{}])[0].get("method", ""),
                "organism": entry.get("rcsb_entity_source_organism", [{}])[0].get("common_name", ""),
                "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date", "")
            }
        return None
    except Exception:
        return None


def _safe_resname(x):
    return (x or "").strip()

def _collect_removable_ligands(atoms, exclude_glycans=True, exclude_cofactors=True):
    excl = EXCLUDE_IONS | HEME_RESNAMES | METAL_RESNAMES
    if exclude_glycans:
        excl |= GLYCAN_NAMES
    if exclude_cofactors:
        excl |= COFACTOR_NAMES
    het = atoms.select("hetatm and not water")
    if het is None:
        return []
    results = []
    for res in het.getHierView().iterResidues():
        rn = _safe_resname(res.getResname()).upper()
        if rn in excl or res.numAtoms() <= _MIN_LIG_ATOMS:
            continue
        if _BACKBONE.issubset(set(res.getNames())):
            continue
        ch = (res.getChid() or "").strip()
        ri = res.getResnum()
        sel = (f"resname {rn} and resid {ri} and chain {ch}" if ch else f"resname {rn} and resid {ri}")
        lig_atoms = atoms.select(sel)
        if lig_atoms is None or lig_atoms.numAtoms() == 0:
            continue
        cx, cy, cz = (float(v) for v in calcCenter(lig_atoms))
        results.append({
            "resname": rn, "chain": ch, "resid": ri,
            "n_atoms": lig_atoms.numAtoms(),
            "cx": cx, "cy": cy, "cz": cz, "sel_str": sel
        })
    results.sort(key=lambda d: (-d["n_atoms"], d["chain"] != "A"))
    return results

def detect_cocrystal_ligand(pdb_path):
    if not PRODY_AVAILABLE:
        return {"found": False, "error": "ProDy not installed"}
    try:
        atoms = parsePDB(pdb_path)
        if atoms is None:
            return {"found": False, "error": "Could not parse PDB"}
        candidates = _collect_removable_ligands(atoms)
        if not candidates:
            return {"found": False, "error": "No drug-like ligand found"}
        best = candidates[0]
        return {
            "found": True,
            "resname": best["resname"],
            "chain": best["chain"],
            "resid": best["resid"],
            "n_atoms": best["n_atoms"],
            "center": (best["cx"], best["cy"], best["cz"]),
            "message": f"{best['resname']} (chain {best['chain']}, resid {best['resid']}, {best['n_atoms']} atoms)",
            "all_candidates": candidates,
        }
    except Exception as e:
        return {"found": False, "error": str(e)}

def detect_chains_and_cofactors(pdb_path):
    """Detect distinct chain IDs and cofactor-like heteroatom residues (e.g. HEM,
    ATP, FAD, buffer components) present in a receptor PDB, so the user can choose
    which chain to keep and which cofactor(s) to retain rather than stripping all
    of them automatically."""
    out = {"chains": [], "cofactors": []}
    if not PRODY_AVAILABLE:
        return out
    try:
        atoms = parsePDB(pdb_path)
        if atoms is None:
            return out
        prot_sel = atoms.select("protein")
        chid_source = prot_sel if prot_sel is not None else atoms
        chains = sorted({c.strip() for c in chid_source.getChids() if c.strip()})
        out["chains"] = chains

        cofactor_like = COFACTOR_NAMES | HEME_RESNAMES
        het = atoms.select("hetatm and not water")
        if het is not None:
            seen = set()
            cofactors = []
            for res in het.getHierView().iterResidues():
                rn = _safe_resname(res.getResname()).upper()
                if rn not in cofactor_like:
                    continue
                ch = (res.getChid() or "").strip()
                ri = res.getResnum()
                key = (rn, ch, ri)
                if key in seen:
                    continue
                seen.add(key)
                cofactors.append({"resname": rn, "chain": ch, "resid": ri})
            cofactors.sort(key=lambda d: (d["resname"], d["chain"], d["resid"]))
            out["cofactors"] = cofactors
        return out
    except Exception:
        return out


VINA_SEARCH_PATHS = {
    "VINA":    ["/content/vina",    "./vina",    "vina",    "/usr/local/bin/vina"],
    "VINAXB":  ["/content/vinaXB",  "./vinaXB",  "vinaXB",  "./vina_xb"],
    "GNINA":   ["/content/gnina",   "./gnina",   "gnina",   "/usr/local/bin/gnina"],
    "UNIDOCK": ["/content/unidock", "./unidock", "unidock", "/usr/local/bin/unidock"],
}

def _file_is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK) and os.path.getsize(path) > 1000

def find_docking_binary(engine, custom=""):
    engine = (engine or "").upper().strip()
    if custom and custom.strip():
        cand = custom.strip()
        if engine == "GNINA" and cand.startswith("docker:"):
            ok, msg = docker_is_ready()
            if ok:
                return cand, cand
            raise RuntimeError(f"Docker GNINA requested but Docker is not ready: {msg}")
        if _file_is_executable(cand):
            return cand, "custom"
        raise RuntimeError(f"Custom binary not found or not executable: {cand}")
    for cand in VINA_SEARCH_PATHS.get(engine, []):
        if _file_is_executable(cand):
            return cand, cand
    if engine == "GNINA":
        ok, msg = docker_is_ready()
        if ok:
            return "docker:gnina/gnina", "Docker image gnina/gnina"
    return None, None

def check_obabel():
    return shutil.which("obabel") is not None

def docker_is_ready():
    if shutil.which("docker") is None:
        return False, "Docker command not found"
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return True, "Docker is running"
        return False, (r.stderr or r.stdout or "Docker is not running").strip()
    except Exception as e:
        return False, str(e)

def _is_docker_gnina(dock_bin):
    return isinstance(dock_bin, str) and dock_bin.startswith("docker:")

def _docker_work_path(path):
    abs_path = os.path.abspath(str(path))
    cwd = os.path.abspath(os.getcwd())
    try:
        rel = os.path.relpath(abs_path, cwd)
        if not rel.startswith(".."):
            return "/work/" + rel.replace(os.sep, "/")
    except Exception:
        pass
    return "/work/" + os.path.basename(abs_path)


def prepare_receptor_vina_batch(raw_pdb, workdir, cx, cy, cz, sx, sy, sz,
                                 keep_chain=None, keep_cofactors=None):
    workdir = Path(workdir)
    log = []
    receptor_pdb = str(workdir / "receptor_atoms.pdb")
    rec_nometal  = str(workdir / "receptor_nometal.pdb")
    receptor_fh  = str(workdir / "rec.pdb")
    receptor_pdbqt = str(workdir / "rec.pdbqt")
    config_file  = str(workdir / "config.txt")
    box_pdb      = str(workdir / "rec.box.pdb")

    keep_cofactors = set(keep_cofactors or [])
    
    if PRODY_AVAILABLE:
        try:
            atoms = parsePDB(raw_pdb)
            all_lig = _collect_removable_ligands(atoms)
            excl_selectors = [d["sel_str"] for d in all_lig]

            cofactor_like = COFACTOR_NAMES | HEME_RESNAMES
            het = atoms.select("hetatm and not water")
            if het is not None:
                for res in het.getHierView().iterResidues():
                    rn = _safe_resname(res.getResname()).upper()
                    if rn not in cofactor_like:
                        continue
                    ch = (res.getChid() or "").strip()
                    ri = res.getResnum()
                    if (rn, ch, ri) in keep_cofactors:
                        continue
                    sel = (f"resname {rn} and resid {ri} and chain {ch}" if ch else f"resname {rn} and resid {ri}")
                    excl_selectors.append(sel)
                    log.append(f"✓ Stripped cofactor {rn} (chain {ch or '-'}, resid {ri})")

            if excl_selectors:
                excl_expr = " or ".join(f"({s})" for s in excl_selectors)
                sel_str = f"not ({excl_expr}) and not water"
            else:
                sel_str = "not water"

            if keep_chain and keep_chain != "All chains":
                sel_str = f"({sel_str}) and chain {keep_chain}"
                log.append(f"✓ Restricted receptor to chain {keep_chain}")

            rec_atoms = atoms.select(sel_str)
            if rec_atoms is not None and rec_atoms.numAtoms() > 0:
                writePDB(receptor_pdb, rec_atoms)
                log.append(f"✓ Receptor selected ({rec_atoms.numAtoms()} atoms)")
            else:
                shutil.copy(raw_pdb, receptor_pdb)
                log.append("⚠ ProDy selection returned no atoms — using raw PDB")
        except Exception as e:
            shutil.copy(raw_pdb, receptor_pdb)
            log.append(f"⚠ ProDy error ({e}) — using raw PDB")
    else:
        shutil.copy(raw_pdb, receptor_pdb)
        log.append("ℹ ProDy not available — using raw PDB as receptor")
    
    metal_lines, clean_lines = [], []
    with open(receptor_pdb) as f:
        for line in f:
            field = line[:6].strip()
            if field in ("ATOM","HETATM") and line[17:20].strip().upper() in METAL_RESNAMES:
                metal_lines.append(line)
            else:
                clean_lines.append(line)
    with open(rec_nometal, "w") as f:
        f.writelines(clean_lines)
    if metal_lines:
        log.append(f"⚠ Stripped {len(metal_lines)} metal atoms before OpenBabel")
    
    r = subprocess.run(f'obabel "{rec_nometal}" -O "{receptor_fh}" -p 7.4 2>/dev/null', shell=True, capture_output=True, text=True)
    if not os.path.exists(receptor_fh) or os.path.getsize(receptor_fh) < 100:
        raise RuntimeError("Hydrogen addition failed (obabel)")
    log.append("✓ Polar hydrogens added to receptor (pH 7.4)")

    rec_h_nometal = str(workdir / "receptor_h_nometal.pdb")
    shutil.copy(receptor_fh, rec_h_nometal)

    if metal_lines:
        lines = [l for l in open(receptor_fh).readlines() if l.strip() != "END"]
        lines.extend(metal_lines)
        lines.append("END\n")
        with open(receptor_fh, "w") as f:
            f.writelines(lines)

    r = subprocess.run(f'obabel "{rec_h_nometal}" -O "{receptor_pdbqt}" -xr --partialcharge gasteiger 2>/dev/null', shell=True, capture_output=True, text=True)
    if not os.path.exists(receptor_pdbqt) or os.path.getsize(receptor_pdbqt) < 100:
        raise RuntimeError("PDBQT conversion failed (obabel)")
    log.append("✓ Receptor converted to PDBQT")
    
    if metal_lines:
        pdbqt_lines = [l for l in open(receptor_pdbqt).readlines() if l.strip() != "END"]
        injected = 0
        for ml in metal_lines:
            try:
                resname = ml[17:20].strip().upper()
                if resname in _NO_REINJECT:
                    continue
                serial = int(ml[6:11])
                name = ml[12:16].strip()
                chain = ml[21] if len(ml) > 21 else "A"
                resid = int(ml[22:26])
                x, y, z = float(ml[30:38]), float(ml[38:46]), float(ml[46:54])
                charge = METAL_CHARGES.get(resname, 0.0)
                atype = resname.capitalize()
                pdbqt_lines.append(f"HETATM{serial:5d} {name:<4s} {resname:<3s} {chain}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {charge:+.3f} {atype}\n")
                injected += 1
            except Exception:
                pass
        pdbqt_lines.append("END\n")
        with open(receptor_pdbqt, "w") as f:
            f.writelines(pdbqt_lines)
        if injected:
            log.append(f"✓ Re-injected {injected} metal atoms into PDBQT")
    
    write_box_pdb(box_pdb, cx, cy, cz, sx, sy, sz)
    write_vina_config(config_file, cx, cy, cz, sx, sy, sz)
    log.append(f"✓ Box: {sx}×{sy}×{sz} Å at ({cx:.2f},{cy:.2f},{cz:.2f})")
    log.append(f"✓ Config: {config_file}")
    
    try:
        rec_lines = open(receptor_fh).readlines()
        coord_lines = [l for l in rec_lines if l[:6].strip() in ("ATOM","HETATM")]
        all_blank = all((l[21]==" " if len(l)>21 else True) for l in coord_lines)
        if all_blank and coord_lines:
            fixed = []
            for l in rec_lines:
                if l[:6].strip() in ("ATOM","HETATM") and len(l) > 21:
                    l = l[:21] + "A" + l[22:]
                fixed.append(l)
            with open(receptor_fh, "w") as f:
                f.writelines(fixed)
            log.append("✓ Assigned chain A to blank-chain atoms")
    except Exception:
        pass
    
    return {
        "success": True,
        "raw_pdb": str(raw_pdb),
        "receptor_pdb": receptor_fh,
        "receptor_pdbqt": receptor_pdbqt,
        "config_file": config_file,
        "box_pdb": box_pdb,
        "log": log
    }


def ph_adjust_smiles(smiles_str, ph=7.4):
    if DIMORPHITE_AVAILABLE:
        try:
            prot_list = _dimorphite_protonate(smiles_str, ph_min=ph, ph_max=ph, max_variants=4)
            candidates = []
            for smi in prot_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
                net = int(sum(charges))
                candidates.append((smi, net))
            if candidates:
                candidates.sort(key=lambda x: abs(x[1]))
                return candidates[0][0]
        except Exception:
            pass
    return smiles_str

def build_3d_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:
        params = AllChem.ETKDG()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol

def write_pdbqt_meeko(mol3d, out_pdbqt):
    if not MEEKO_AVAILABLE:
        raise RuntimeError("Meeko not available")
    mol3d = Chem.Mol(mol3d)
    if any(a.GetNumImplicitHs() > 0 for a in mol3d.GetAtoms()):
        mol3d = Chem.AddHs(mol3d, addCoords=True)
    prep = MoleculePreparation()
    if MEEKO_LEGACY:
        setups = prep.prepare(mol3d)
        pdbqt_str, _, _ = _MeekoWriter.write_string(setups[0])
    else:
        prep.prepare(mol3d)
        pdbqt_str = prep.write_pdbqt_string()
    with open(out_pdbqt, "w") as f:
        f.write(pdbqt_str)
    assert_file_ok(out_pdbqt, 100, f"PDBQT write failed: {out_pdbqt}")

def prepare_ligand_vina_batch(smiles, lig_name, ph, workdir):
    workdir = Path(workdir)
    log = []
    prefix = safe_name(lig_name)
    out_pdb   = str(workdir / f"{prefix}_min.pdb")
    out_sdf   = str(workdir / f"{prefix}_min.sdf")
    out_pdbqt = str(workdir / f"{prefix}_min.pdbqt")
    
    ph_smiles = ph_adjust_smiles(smiles, ph)
    log.append(f"✓ SMILES @ pH {ph}: {ph_smiles[:60]}")
    mol = build_3d_mol(ph_smiles)
    log.append(f"✓ 3D conformer: {mol.GetNumAtoms()} atoms")
    Chem.MolToPDBFile(mol, out_pdb)
    with Chem.SDWriter(out_sdf) as w:
        w.write(mol)
    log.append(f"✓ PDB: {out_pdb}")
    log.append(f"✓ SDF: {out_sdf}")
    
    try:
        write_pdbqt_meeko(mol, out_pdbqt)
        log.append(f"✓ PDBQT (meeko): {out_pdbqt}")
    except Exception as e:
        log.append(f"⚠ Meeko failed ({e}) — trying obabel fallback")
        r = subprocess.run(f'obabel "{out_sdf}" -O "{out_pdbqt}" -xh 2>/dev/null', shell=True, capture_output=True, text=True)
        if not os.path.exists(out_pdbqt) or os.path.getsize(out_pdbqt) < 100:
            raise RuntimeError(f"PDBQT conversion failed for {lig_name}")
        log.append(f"✓ PDBQT (obabel): {out_pdbqt}")
    
    return {
        "success": True,
        "name": lig_name,
        "pdb": out_pdb,
        "sdf": out_sdf,
        "pdbqt": out_pdbqt,
        "ph_smiles": ph_smiles,
        "log": log
    }


def validate_ligand_preparation_with_rmsd(smiles_string, prepared_sdf_path, reference_pdb=None):

    try:
        template = Chem.MolFromSmiles(smiles_string)
        prepared_mol = _first_sdf_mol(prepared_sdf_path)
        
        if template is None or prepared_mol is None:
            return None
        
        prepared_mol = _assign_template_bonds(template, prepared_mol)
        query = _heavy_mol(template) if template is not None else None
        
        if reference_pdb and os.path.exists(reference_pdb):
            ref_mol = Chem.MolFromPDBFile(reference_pdb, sanitize=False, removeHs=False, proximityBonding=True)
            if ref_mol:
                ref_mol = _assign_template_bonds(template, ref_mol)
                return _rmsd_against_reference(ref_mol, prepared_mol, query)
        
        return {"heavy_atoms": template.GetNumHeavyAtoms(), "conformer_energy": None}
    
    except Exception as e:
        return None

def display_rmsd_validation_ui():

    st.markdown("---")
    st.markdown("### RMSD Validation")
    st.markdown("""
    > **Purpose:** Validate your ligand preparation by comparing the generated 3D structure
    > with an experimental reference (if available) or by checking structural integrity.
    > This helps ensure that your ligands are correctly prepared before batch docking.
    """)
    
    col_val1, col_val2 = st.columns(2)
    
    with col_val1:
        st.markdown("#### Quick Validation")
        st.markdown("""
        - Checks if SMILES can be converted to 3D structure
        - Verifies atom count and connectivity
        - No reference structure required
        """)
        
        if st.button("Validate Ligands", key="validate_ligands_btn", use_container_width=True):
            smiles_list = st.session_state.get("selected_smiles_list", [])
            if not smiles_list:
                st.warning("No ligands selected for validation.")
            else:
                results = []
                for name, smiles in smiles_list[:10]:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            heavy_atoms = mol.GetNumHeavyAtoms()
                            mol_3d = build_3d_mol(smiles)
                            results.append({
                                "Name": name,
                                "SMILES": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                                "Heavy Atoms": heavy_atoms,
                                "Valid": "✅",
                                "3D Generated": "✅" if mol_3d else "❌"
                            })
                        else:
                            results.append({
                                "Name": name,
                                "SMILES": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                                "Heavy Atoms": "N/A",
                                "Valid": "❌",
                                "3D Generated": "❌"
                            })
                    except Exception as e:
                        results.append({
                            "Name": name,
                            "SMILES": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                            "Heavy Atoms": "N/A",
                            "Valid": "❌",
                            "3D Generated": f"Error: {str(e)[:30]}"
                        })
                
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                
                valid_count = sum(1 for r in results if r["Valid"] == "✅")
                st.success(f"✅ {valid_count}/{len(results)} ligands passed validation")
                
                if valid_count < len(results):
                    st.warning("Some ligands failed validation. Check SMILES strings for errors.")
    
    with col_val2:
        st.markdown("#### RMSD Calculator")
        st.markdown("""
        For redocking validation, use the **Redocking Validation** section above.
        This tool is for comparing prepared ligands against reference structures.
        """)
        
        ref_file = st.file_uploader(
            "Upload reference ligand structure (PDB or SDF)",
            type=["pdb", "sdf"],
            key="rmsd_ref_upload",
            help="Optional: Upload experimental reference to calculate RMSD"
        )
        
        if ref_file:
            ref_path = unique_filepath(os.path.join(OUTPUT_DIR, f"rmsd_ref.{ref_file.name.split('.')[-1]}"))
            with open(ref_path, "wb") as f:
                f.write(ref_file.getvalue())
            
            st.success(f"✅ Reference uploaded: {ref_file.name}")
            
            smiles_list = st.session_state.get("selected_smiles_list", [])
            if smiles_list:
                lig_options = [f"{name}" for name, _ in smiles_list[:20]]
                selected_lig = st.selectbox("Select ligand to compare", lig_options, key="rmsd_lig_select")
                
                if selected_lig and st.button("Calculate RMSD", key="calc_rmsd_btn"):
                    lig_smiles = next((smi for name, smi in smiles_list if name == selected_lig), None)
                    if lig_smiles:
                        temp_dir = unique_workdir(Path(OUTPUT_DIR) / "rmsd_temp")
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            prep_result = prepare_ligand_vina_batch(lig_smiles, "temp", 7.4, temp_dir)
                            if prep_result['success']:
                                rmsd_val = calculate_rmsd_between_ligands(
                                    ref_path, prep_result['sdf'], "LIG", lig_smiles
                                )
                                if rmsd_val is not None:
                                    message, status = get_rmsd_validation_message(rmsd_val)
                                    st.markdown(f"""
                                    <div class="rmsd-card">
                                        <strong>RMSD Calculation Result:</strong><br>
                                        <span class="rmsd-value">{rmsd_val:.3f} Å</span><br>
                                        {message}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning("Could not calculate RMSD. Check atom matching.")
                            else:
                                st.error("Failed to prepare ligand for comparison.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                        finally:
                            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_vina_config_box(config_path):
    want = {k: None for k in ("center_x","center_y","center_z","size_x","size_y","size_z")}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for k in want:
                if line.lower().startswith(k):
                    parts = re.split(r"[=\s]+", line, maxsplit=1)
                    if len(parts) >= 2:
                        try:
                            want[k] = float(parts[1])
                        except ValueError:
                            pass
    missing = [k for k, v in want.items() if v is None]
    if missing:
        raise RuntimeError(f"Could not parse {missing} from config: {config_path}")
    return (want["center_x"], want["center_y"], want["center_z"], want["size_x"], want["size_y"], want["size_z"])

def convert_pdbqt_to_sdf(pdbqt_file, sdf_file=None):
    sdf_file = sdf_file or pdbqt_file.replace(".pdbqt", ".sdf")
    r = subprocess.run(["obabel", pdbqt_file, "-O", sdf_file], capture_output=True, text=True, timeout=60)
    if not os.path.exists(sdf_file) or os.path.getsize(sdf_file) < 10:
        raise RuntimeError(f"SDF conversion failed: {sdf_file}")
    return sdf_file

def _find_unidock_outputs(out_dir):
    if not os.path.isdir(out_dir):
        return {"pdbqt": None, "sdf": None, "all": []}
    all_files = sorted([p for p in glob.glob(os.path.join(out_dir,"*")) if os.path.isfile(p) and os.path.getsize(p)>0])
    sdf_hits = sorted(glob.glob(os.path.join(out_dir,"*_out*.sdf"))) + sorted(glob.glob(os.path.join(out_dir,"*.sdf")))
    pdbqt_hits = sorted(glob.glob(os.path.join(out_dir,"*_out*.pdbqt"))) + sorted(glob.glob(os.path.join(out_dir,"*.pdbqt")))
    def _uniq(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen and os.path.isfile(x) and os.path.getsize(x)>10:
                seen.add(x); out.append(x)
        return out
    sdf_hits = _uniq(sdf_hits); pdbqt_hits = _uniq(pdbqt_hits)
    return {"sdf": sdf_hits[0] if sdf_hits else None, "pdbqt": pdbqt_hits[0] if pdbqt_hits else None, "all": all_files}

def _update_live_log(log_placeholder, lines, header=""):
    if log_placeholder is None:
        return
    tail = "".join(lines[-80:]).strip()
    body = (header + "\n\n" if header else "") + (tail or "Running...")
    log_placeholder.code(body[-12000:], language="text")

def run_subprocess_docking(cmd_list, timeout=1800, log_placeholder=None):
    header = "$ " + " ".join(str(x) for x in cmd_list)
    if log_placeholder is None:
        try:
            r = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return -1, f"Timed out after {timeout} seconds"
        except Exception as e:
            return -1, str(e)

    lines = []
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        _update_live_log(log_placeholder, lines, header)
        while True:
            if proc.poll() is not None:
                rest = proc.stdout.read() if proc.stdout else ""
                if rest:
                    lines.append(rest)
                break
            if time.time() - start > timeout:
                proc.kill()
                lines.append(f"\nTimed out after {timeout} seconds\n")
                _update_live_log(log_placeholder, lines, header)
                return -1, "".join(lines)
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                lines.append(line)
                _update_live_log(log_placeholder, lines, header)
            else:
                time.sleep(0.2)
        _update_live_log(log_placeholder, lines, header)
        return proc.returncode, "".join(lines)
    except subprocess.TimeoutExpired:
        return -1, f"Timed out after {timeout} seconds"
    except Exception as e:
        return -1, str(e)

def dock_one_ligand(engine, dock_bin, receptor_pdbqt, ligand_input, config_file, out_prefix, params):
    log = ""
    timeout = int(params.get("timeout", 1800))
    log_placeholder = params.get("log_placeholder")
    
    if engine == "GNINA":
        cx, cy, cz, sx, sy, sz = parse_vina_config_box(config_file)
        out_sdf = f"{out_prefix}.sdf"
        gnina_args = [
            "-r", receptor_pdbqt, "-l", ligand_input,
            "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
            "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
            "--exhaustiveness", str(params.get("gnina_exhaustiveness", 8)),
            "--num_modes", str(params.get("gnina_num_modes", 9)),
            "--cnn_scoring", params.get("gnina_cnn_scoring", "rescore"),
            "-o", out_sdf
        ]
        if _is_docker_gnina(dock_bin):
            image = dock_bin.split(":", 1)[1] or "gnina/gnina"
            docker_args = [
                "-r", _docker_work_path(receptor_pdbqt), "-l", _docker_work_path(ligand_input),
                "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                "--exhaustiveness", str(params.get("gnina_exhaustiveness", 8)),
                "--num_modes", str(params.get("gnina_num_modes", 9)),
                "--cnn_scoring", params.get("gnina_cnn_scoring", "rescore"),
                "-o", _docker_work_path(out_sdf)
            ]
            cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{os.getcwd()}:/work", "-w", "/work", image, "gnina"] + docker_args
        else:
            cmd = [dock_bin] + gnina_args
        rc, log = run_subprocess_docking(cmd, timeout=timeout, log_placeholder=log_placeholder)
        if rc != 0 or not os.path.exists(out_sdf) or os.path.getsize(out_sdf) < 10:
            raise RuntimeError(f"GNINA failed (exit {rc}): {log[:500]}")
        return None, out_sdf, log
    
    if engine == "UNIDOCK":
        cx, cy, cz, sx, sy, sz = parse_vina_config_box(config_file)
        out_dir = out_prefix + "_unidock_out"
        os.makedirs(out_dir, exist_ok=True)
        search_mode = params.get("unidock_search_mode", "balance")
        cmd = [
            dock_bin, "--receptor", receptor_pdbqt, "--gpu_batch", ligand_input,
            "--scoring", params.get("unidock_scoring", "vina"),
            "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
            "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
            "--num_modes", str(params.get("unidock_num_modes", 9)),
            "--dir", out_dir,
            "--seed", str(params.get("unidock_seed", 0))
        ]
        if search_mode != "custom":
            cmd += ["--search_mode", search_mode]
        else:
            cmd += [
                "--exhaustiveness", str(params.get("unidock_exhaustiveness", 512)),
                "--max_step", str(params.get("unidock_max_step", 40))
            ]
        rc, log = run_subprocess_docking(cmd, timeout=timeout, log_placeholder=log_placeholder)
        hits = _find_unidock_outputs(out_dir)
        docked_sdf, docked_pdbqt = hits["sdf"], hits["pdbqt"]
        if docked_sdf is None and docked_pdbqt is not None:
            docked_sdf = convert_pdbqt_to_sdf(docked_pdbqt)
        if docked_sdf is None and docked_pdbqt is None:
            raise RuntimeError(f"Uni-Dock produced no output in {out_dir}: {log[:500]}")
        return docked_pdbqt, docked_sdf, log
    
    out_pdbqt = f"{out_prefix}.pdbqt"
    cmd = [
        dock_bin, "--receptor", receptor_pdbqt, "--ligand", ligand_input,
        "--config", config_file,
        "--exhaustiveness", str(params.get("exhaustiveness", 16)),
        "--num_modes", str(params.get("num_modes", 9)),
        "--energy_range", str(params.get("energy_range", 3)),
        "--out", out_pdbqt
    ]
    rc, log = run_subprocess_docking(cmd, timeout=timeout, log_placeholder=log_placeholder)
    if not os.path.exists(out_pdbqt) or os.path.getsize(out_pdbqt) < 10:
        raise RuntimeError(f"{engine} failed or produced no output (exit {rc}): {log[:500]}")
    out_sdf = convert_pdbqt_to_sdf(out_pdbqt)
    return out_pdbqt, out_sdf, log


def extract_top_score(pdbqt_path, sdf_path, engine):
    engine = engine.upper()
    if engine == "GNINA":
        if sdf_path and os.path.exists(sdf_path):
            supp = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
            mol = next((m for m in supp if m is not None), None)
            if mol:
                for prop in ("minimizedAffinity","affinity","docking_score"):
                    if mol.HasProp(prop):
                        return {
                            "score": float(mol.GetProp(prop)),
                            "CNNscore": float(mol.GetProp("CNNscore")) if mol.HasProp("CNNscore") else None,
                            "CNNaffinity": float(mol.GetProp("CNNaffinity")) if mol.HasProp("CNNaffinity") else None
                        }
        raise ValueError("No GNINA affinity property found")
    
    if engine == "UNIDOCK":
        if sdf_path and os.path.exists(sdf_path):
            with open(sdf_path) as f:
                for line in f:
                    if "ENERGY=" in line:
                        nums = re.findall(r"[-+]?\d*\.?\d+", line)
                        if nums:
                            return {
                                "score": float(nums[0]),
                                "rmsd_lb": float(nums[1]) if len(nums)>=2 else "",
                                "rmsd_ub": float(nums[2]) if len(nums)>=3 else ""
                            }
        raise ValueError("No UNIDOCK ENERGY= line found")
    
    if pdbqt_path and os.path.exists(pdbqt_path):
        with open(pdbqt_path) as f:
            for line in f:
                if re.search(r"VINA(?:XB)? RESULT", line):
                    parts = line.split()
                    try:
                        return {
                            "score": float(parts[3]),
                            "rmsd_lb": float(parts[4]),
                            "rmsd_ub": float(parts[5])
                        }
                    except:
                        pass
    raise ValueError(f"No VINA RESULT line found in {pdbqt_path}")

def parse_all_poses(pdbqt_path=None, sdf_path=None, engine="VINA"):
    engine = engine.upper()
    
    if engine == "GNINA" and sdf_path and os.path.exists(sdf_path):
        supp = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        poses = []
        for i, mol in enumerate(supp):
            if mol is None: continue
            aff = None
            for prop in ("minimizedAffinity","affinity"):
                if mol.HasProp(prop):
                    try: aff = float(mol.GetProp(prop)); break
                    except: pass
            poses.append({
                "pose": i+1,
                "affinity": aff,
                "CNNscore": float(mol.GetProp("CNNscore")) if mol.HasProp("CNNscore") else None,
                "CNNaffinity": float(mol.GetProp("CNNaffinity")) if mol.HasProp("CNNaffinity") else None
            })
        return poses
    
    if engine == "UNIDOCK" and sdf_path and os.path.exists(sdf_path):
        with open(sdf_path) as f:
            content = f.read()
        blocks = content.split("$$$$")
        poses = []
        for i, block in enumerate(blocks):
            if not block.strip(): continue
            nums = []
            for line in block.splitlines():
                if "ENERGY=" in line:
                    nums = re.findall(r"[-+]?\d*\.?\d+", line)
                    break
            aff = float(nums[0]) if nums else None
            poses.append({
                "pose": i+1,
                "affinity": aff,
                "rmsd_lb": float(nums[1]) if len(nums)>=2 else None,
                "rmsd_ub": float(nums[2]) if len(nums)>=3 else None
            })
        return poses
    
    poses = []
    if pdbqt_path and os.path.exists(pdbqt_path):
        current_mode = None
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith("MODEL"):
                    try: current_mode = int(line.split()[1])
                    except: pass
                elif re.search(r"VINA(?:XB)? RESULT", line):
                    parts = line.split()
                    try:
                        poses.append({
                            "pose": current_mode,
                            "affinity": float(parts[3]),
                            "rmsd_lb": float(parts[4]),
                            "rmsd_ub": float(parts[5])
                        })
                    except: pass
    return poses


_ADMET_CYP_SMARTS = {
    "CYP1A2":  "[$([nH]1cncc1),$([n+]1cnccc1),$([NH]c1ccc2ccccc2n1)]",
    "CYP2C9":  "[$([SX4](=O)(=O)),$([c;R1]1ccc(cc1)[NX3]),$([CX3](=O)[NX3;H1]c)]",
    "CYP2C19": "[$([nX2]1cccc1),$([NX3;H1][CX3]=O),$([OX2][CX3]=O)]",
    "CYP2D6":  "[$([NX3;H1,H2]Cc1ccccc1),$([NX3]c1ccc[nH]1),$([nH]1cncc1)]",
    "CYP3A4":  "[$([#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~1),$([CX3](=O)[OX2H0])]",
}


@st.cache_resource(show_spinner="Loading ADMET-AI models (first run only)…")
def _load_admet_model_cached():
    """Load the ADMET-AI model once per server process.

    Returns (model, error_string). error_string is None on success, "NOT_INSTALLED"
    if the package simply isn't importable, or the real exception text if the
    package is installed but failed to load (wrong Python version, missing model
    weights, incompatible torch/chemprop, etc.) so the UI can show the true cause
    instead of always saying "not installed".
    """
    try:
        from admet_ai import ADMETModel
    except ImportError as e:
        return None, f"NOT_INSTALLED::{e}"
    try:
        return ADMETModel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _load_admet_model():
    """Backward-compatible wrapper: returns just the model (or None)."""
    model, _ = _load_admet_model_cached()
    return model


def _calc_admet_properties(smiles: str) -> dict:
    """Calculate ADMET properties for a single ligand.

    Part 1: RDKit descriptors + rule-based ADME estimates (always available, offline).
    Part 2: ADMET-AI ML predictions (only used if `pip install admet-ai` is available);
            falls back gracefully to the rule-based estimates otherwise.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, QED
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        mw      = round(Descriptors.MolWt(mol), 2)
        logp    = round(Descriptors.MolLogP(mol), 2)
        hbd     = rdMolDescriptors.CalcNumHBD(mol)
        hba     = rdMolDescriptors.CalcNumHBA(mol)
        tpsa    = round(rdMolDescriptors.CalcTPSA(mol), 2)
        rotb    = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rings   = rdMolDescriptors.CalcNumRings(mol)
        arom    = rdMolDescriptors.CalcNumAromaticRings(mol)
        heavy   = mol.GetNumHeavyAtoms()
        qed_val = round(QED.qed(mol), 3)
        exact_mw = round(rdMolDescriptors.CalcExactMolWt(mol), 4)
        fsp3     = round(rdMolDescriptors.CalcFractionCSP3(mol), 3)

        lip_viol    = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        lip_pass    = lip_viol <= 1
        veber_pass  = (rotb <= 10 and tpsa <= 140)
        egan_pass   = (logp <= 5.88 and tpsa <= 131.6)
        muegge_pass = (
            200 <= mw <= 600 and -2 <= logp <= 5 and tpsa <= 150
            and rings <= 7 and heavy <= 30 and rotb <= 15
            and hbd <= 5 and hba <= 10
        )
        bio_score = round(sum([lip_pass, veber_pass, egan_pass, muegge_pass]) / 4.0, 2)

        cyp_flags = {}
        for cn, sma in _ADMET_CYP_SMARTS.items():
            try:
                pat = Chem.MolFromSmarts(sma)
                cyp_flags[cn] = bool(pat and mol.HasSubstructMatch(pat))
            except Exception:
                cyp_flags[cn] = False

        alerts_pains, alerts_brenk = [], []
        try:
            pp = FilterCatalogParams()
            pp.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            pp.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            pp.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            for e in FilterCatalog(pp).GetMatches(mol):
                alerts_pains.append(e.GetDescription())
        except Exception:
            pass
        try:
            bp = FilterCatalogParams()
            bp.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            for e in FilterCatalog(bp).GetMatches(mol):
                alerts_brenk.append(e.GetDescription())
        except Exception:
            pass

        ml_results, ml_source, ml_error, ml_available = {}, "rule-based (ADMET-AI not installed)", None, False
        try:
            admet_model, load_err = _load_admet_model_cached()
            if admet_model is not None:
                raw = admet_model.predict(smiles=smiles)
                ml_results["bioavailability_ml"] = round(float(raw.get("Bioavailability_Ma", 0)), 3)
                ml_results["bbb_prob"] = round(float(raw.get("BBB_Martins", 0)), 3)
                ml_results["ppb"]      = round(float(raw.get("PPBR_AZ", 0)), 1)
                ml_results["herg"]      = float(raw.get("hERG", 0)) > 0.5
                ml_results["herg_prob"] = round(float(raw.get("hERG", 0)), 3)
                ml_results["ames"]      = float(raw.get("AMES", 0)) > 0.5
                ml_results["ames_prob"] = round(float(raw.get("AMES", 0)), 3)
                ml_results["dili"]      = float(raw.get("DILI", 0)) > 0.5
                ml_results["dili_prob"] = round(float(raw.get("DILI", 0)), 3)
                ml_results["ld50"]      = round(float(raw.get("LD50_Zhu", 0)), 2)
                ml_results["half_life"] = round(float(raw.get("Half_Life_Obach", 0)), 2)
                ml_results["vdss"] = round(float(raw.get("VDss_Lombardo", 0)), 2)
                ml_results["hepatic_clearance"] = round(float(raw.get("Clearance_Hepatocyte_AZ", 0)), 2)
                ml_results["skin_reaction"] = float(raw.get("Skin_Reaction_Martins", 0)) > 0.5
                ml_results["skin_reaction_prob"] = round(float(raw.get("Skin_Reaction_Martins", 0)), 3)
                ml_results["pgp_substrate_ml"] = float(raw.get("Pgp_Broccatelli", 0)) > 0.5
                for _cn, _col in [("CYP1A2", "CYP1A2_Veith"), ("CYP2C9", "CYP2C9_Veith"),
                                   ("CYP2C19", "CYP2C19_Veith"), ("CYP2D6", "CYP2D6_Veith"),
                                   ("CYP3A4", "CYP3A4_Veith")]:
                    ml_results[f"{_cn.lower()}_inhib_prob"] = round(float(raw.get(_col, 0)), 3)
                for _cn, _col in [("CYP2C9", "CYP2C9_Substrate_CarbonMangels"),
                                   ("CYP2D6", "CYP2D6_Substrate_CarbonMangels"),
                                   ("CYP3A4", "CYP3A4_Substrate_CarbonMangels")]:
                    ml_results[f"{_cn.lower()}_sub_prob"] = round(float(raw.get(_col, 0)), 3)
                ml_source = "ADMET-AI (Chemprop MPNN + TDC datasets)"
                ml_available = True
            elif load_err:
                if load_err.startswith("NOT_INSTALLED::"):
                    ml_source = "rule-based (ADMET-AI not installed)"
                else:
                    ml_error = load_err
                    ml_source = f"rule-based (ADMET-AI failed to load: {load_err})"
        except Exception as _ml_e:
            ml_error = str(_ml_e)
            ml_source = f"rule-based (ADMET-AI error: {_ml_e})"

        if "bbb_prob" in ml_results:
            bbb = ("Penetrant" if ml_results["bbb_prob"] > 0.7
                   else "Possible" if ml_results["bbb_prob"] > 0.4
                   else "Non-penetrant")
        else:
            bbb_pts = ((1 if 1 <= logp <= 3 else 0) + (1 if tpsa <= 90 else 0)
                       + (1 if mw <= 450 else 0) + (1 if hbd <= 3 else 0) + (1 if rings <= 4 else 0))
            bbb = "Penetrant" if bbb_pts >= 4 else ("Possible" if bbb_pts >= 2 else "Non-penetrant")

        if ml_results.get("pgp_substrate_ml") is not None:
            pgp = "Likely" if ml_results["pgp_substrate_ml"] else "Unlikely"
        else:
            pgp = "Likely" if (mw > 400 and (hba > 4 or rotb > 10)) else "Unlikely"

        if "bioavailability_ml" in ml_results:
            gi = ("High" if ml_results["bioavailability_ml"] > 0.7
                  else "Medium" if ml_results["bioavailability_ml"] > 0.4 else "Low")
        else:
            gi = ("High" if (tpsa <= 131.6 and logp <= 5.88)
                  else "Low" if (tpsa > 200 or logp > 7) else "Medium")

        return {
            "mw": mw, "exact_mw": exact_mw, "logp": logp, "hbd": hbd, "hba": hba, "tpsa": tpsa,
            "rotb": rotb, "rings": rings, "arom_rings": arom, "heavy_atoms": heavy, "fsp3": fsp3,
            "qed": qed_val, "lipinski_violations": lip_viol, "lipinski_pass": lip_pass,
            "veber_pass": veber_pass, "egan_pass": egan_pass, "muegge_pass": muegge_pass,
            "bioavailability_score": bio_score, "gi_absorption": gi, "bbb": bbb,
            "pgp_substrate": pgp, "cyp_flags": cyp_flags,
            "pains_alerts": alerts_pains, "brenk_alerts": alerts_brenk,
            "ml_results": ml_results, "ml_source": ml_source,
            "ml_error": ml_error, "ml_available": ml_available,
        }
    except Exception as e:
        return {"error": str(e)}


_ADMET_STATUS_COLOR = {"pass": "#2e7d32", "warn": "#9A6700", "fail": "#c62828", None: "#24292F"}


def _admet_stat_html(label, value, unit="", status=None):
    clr = _ADMET_STATUS_COLOR.get(status, _ADMET_STATUS_COLOR[None])
    return (
        f'<div class="admet-stat"><div class="admet-stat-label">{label}</div>'
        f'<div class="admet-stat-value" style="color:{clr};">{value}'
        f'<span class="admet-stat-unit">{unit}</span></div></div>'
    )


def _admet_chip_html(name, passed, note=""):
    if passed:
        bg, border, clr = "#e8f5e9", "#2e7d32", "#1b5e20"
    else:
        bg, border, clr = "#fdecea", "#c62828", "#c62828"
    mark = "●" if passed else "○"
    return (
        f'<span class="admet-chip" style="background:{bg};border-color:{border};color:{clr};">'
        f'{mark} {name}{f" · {note}" if note else ""}</span>'
    )


def _render_admet_compound(props, name, score):
    """Render one compound's ADMET panel (used inside a per-compound expander)."""
    if "error" in props:
        st.error(f"ADMET calculation error for {name}: {props['error']}")
        return

    ml_avail = props.get("ml_available", False)
    ml_src   = props.get("ml_source", "")
    src_bg, src_border, src_clr = ("#e8f5e9", "#2e7d32", "#1b5e20") if ml_avail else ("#fff8e1", "#9A6700", "#6b5200")
    src_icon = "◆" if ml_avail else "◇"
    st.markdown(
        f'<div class="admet-source-tag" style="background:{src_bg};border:1px solid {src_border};color:{src_clr};">'
        f'{src_icon} {ml_src}</div>',
        unsafe_allow_html=True,
    )

    stats = [
        ("MW", props["mw"], " g/mol", "pass" if props["mw"] <= 500 else ("warn" if props["mw"] <= 600 else "fail")),
        ("Exact MW", props["exact_mw"], " Da", None),
        ("LogP", props["logp"], "", "pass" if -1 <= props["logp"] <= 3 else ("warn" if props["logp"] <= 5 else "fail")),
        ("TPSA", props["tpsa"], " Å²", "pass" if props["tpsa"] <= 90 else ("warn" if props["tpsa"] <= 140 else "fail")),
        ("HBD", props["hbd"], "", "pass" if props["hbd"] <= 3 else ("warn" if props["hbd"] <= 5 else "fail")),
        ("HBA", props["hba"], "", "pass" if props["hba"] <= 7 else ("warn" if props["hba"] <= 10 else "fail")),
        ("Rot. Bonds", props["rotb"], "", "pass" if props["rotb"] <= 7 else ("warn" if props["rotb"] <= 10 else "fail")),
        ("QED", props["qed"], "", "pass" if props["qed"] >= 0.5 else ("warn" if props["qed"] >= 0.3 else "fail")),
        ("Fsp³", props["fsp3"], "", "pass" if props["fsp3"] >= 0.42 else ("warn" if props["fsp3"] >= 0.25 else "fail")),
        ("Arom Rings", props["arom_rings"], "", None),
        ("Heavy Atoms", props["heavy_atoms"], "", None),
        ("Rings", props["rings"], "", None),
    ]
    st.markdown(
        '<div class="admet-strip">' + "".join(_admet_stat_html(*s) for s in stats) + '</div>',
        unsafe_allow_html=True,
    )

    _bio = int(props["bioavailability_score"] * 100)
    chips_html = (
        _admet_chip_html("Lipinski RO5", props["lipinski_pass"], f"{props['lipinski_violations']} viol.")
        + _admet_chip_html("Veber", props["veber_pass"])
        + _admet_chip_html("Egan", props["egan_pass"])
        + _admet_chip_html("Muegge", props["muegge_pass"])
    )
    st.markdown(f'<div style="margin:10px 0 2px;">{chips_html}</div>', unsafe_allow_html=True)
    st.caption(f"Bioavailability score {_bio}%")

    pa, pb, pc = st.columns(3)
    for col, lbl, val, clr_map in [
        (pa, "GI Absorption", props["gi_absorption"], {"High": "#2e7d32", "Medium": "#9A6700", "Low": "#c62828"}),
        (pb, "BBB Penetration", props["bbb"], {"Penetrant": "#2e7d32", "Possible": "#9A6700", "Non-penetrant": "#c62828"}),
        (pc, "P-gp Substrate", props["pgp_substrate"], {"Unlikely": "#2e7d32", "Likely": "#9A6700"}),
    ]:
        clr = clr_map.get(val, "#888")
        col.markdown(
            f'<div class="admet-verdict-card" style="border-left:3px solid {clr};">'
            f'<div class="admet-verdict-label">{lbl}</div>'
            f'<div class="admet-verdict-value" style="color:{clr};">{val}</div></div>',
            unsafe_allow_html=True,
        )

    ml_res = props.get("ml_results", {})
    st.markdown("")
    st.markdown("**CYP Inhibition Flags**")
    cyp_order = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
    cyp_cols = st.columns(5)
    for ccol, cn in zip(cyp_cols, cyp_order):
        pkey = f"{cn.lower()}_inhib_prob"
        if ml_avail and pkey in ml_res:
            prob = ml_res[pkey]
            flagged = prob > 0.5
        else:
            flagged = props.get("cyp_flags", {}).get(cn, False)
            prob = None
        verdict = "Possible" if flagged else "Unlikely"
        clr = "#9A6700" if flagged else "#2e7d32"
        ccol.markdown(
            f'<div class="admet-verdict-card" style="border-left:3px solid {clr};">'
            f'<div class="admet-verdict-label">{cn}</div>'
            f'<div class="admet-verdict-value" style="color:{clr};font-size:0.85rem;">{verdict}</div>'
            + (f'<div style="font-size:0.62rem;color:#8a8f98;font-family:\'IBM Plex Mono\',monospace;">p = {prob:.3f}</div>'
               if prob is not None else "")
            + '</div>', unsafe_allow_html=True,
        )
    if not ml_avail:
        st.caption("Structural SMARTS flags only — screening purpose, not quantitative.")

    if ml_avail and any(k in ml_res for k in ("cyp2c9_sub_prob", "cyp2d6_sub_prob", "cyp3a4_sub_prob")):
        st.markdown("")
        st.markdown("**CYP Substrate Predictions (ADMET-AI ML)**")
        sub_cols = st.columns(3)
        for scol, cn in zip(sub_cols, ["CYP2C9", "CYP2D6", "CYP3A4"]):
            prob = ml_res.get(f"{cn.lower()}_sub_prob")
            if prob is None:
                continue
            flagged = prob > 0.5
            verdict = "Substrate" if flagged else "Non-substrate"
            clr = "#9A6700" if flagged else "#2e7d32"
            scol.markdown(
                f'<div class="admet-verdict-card" style="border-left:3px solid {clr};">'
                f'<div class="admet-verdict-label">{cn}</div>'
                f'<div class="admet-verdict-value" style="color:{clr};">{verdict}</div>'
                f'<div style="font-size:0.68rem;color:#8a8f98;font-family:\'IBM Plex Mono\',monospace;">p = {prob:.3f}</div>'
                '</div>', unsafe_allow_html=True,
            )

    if ml_avail and ("ppb" in ml_res or "vdss" in ml_res):
        st.markdown("")
        dcol1, dcol2 = st.columns(2)
        if "ppb" in ml_res:
            ppb_val = ml_res["ppb"]
            note = "High — low free fraction" if ppb_val >= 90 else ("Moderate" if ppb_val >= 70 else "Low — high free fraction")
            dcol1.markdown(
                f'<div class="admet-verdict-card" style="border-left:3px solid #556;">'
                f'<div class="admet-verdict-label">Plasma Protein Binding (PPB)</div>'
                f'<div class="admet-verdict-value">{ppb_val}%</div>'
                f'<div style="font-size:0.68rem;color:#8a8f98;">{note}</div></div>',
                unsafe_allow_html=True,
            )
        if "vdss" in ml_res:
            vd_val = ml_res["vdss"]
            note = "Extensive tissue distribution" if vd_val >= 3 else ("Moderate distribution" if vd_val >= 0.7 else "Confined to plasma/blood")
            dcol2.markdown(
                f'<div class="admet-verdict-card" style="border-left:3px solid #556;">'
                f'<div class="admet-verdict-label">Volume of Distribution (VDd)</div>'
                f'<div class="admet-verdict-value">{vd_val} L/kg</div>'
                f'<div style="font-size:0.68rem;color:#8a8f98;">{note}</div></div>',
                unsafe_allow_html=True,
            )

    if ml_avail and ml_res.get("herg") is not None:
        st.markdown("")
        st.markdown("**Toxicity Predictions (ADMET-AI ML)**")
        tc = st.columns(4)
        for col, (lbl, fkey, pkey) in zip(tc, [
            ("hERG inhibition", "herg", "herg_prob"),
            ("AMES mutagenicity", "ames", "ames_prob"),
            ("DILI (liver)", "dili", "dili_prob"),
            ("Skin reaction", "skin_reaction", "skin_reaction_prob"),
        ]):
            flag = ml_res.get(fkey, False)
            prob = ml_res.get(pkey)
            clr = "#c62828" if flag else "#2e7d32"
            verdict = "Positive" if flag else "Negative"
            col.markdown(
                f'<div class="admet-verdict-card" style="border-left:3px solid {clr};">'
                f'<div class="admet-verdict-label">{lbl}</div>'
                f'<div class="admet-verdict-value" style="color:{clr};">{verdict}</div>'
                + (f'<div style="font-size:0.68rem;color:#8a8f98;font-family:\'IBM Plex Mono\',monospace;">p = {prob:.3f}</div>'
                   if prob is not None else "")
                + '</div>', unsafe_allow_html=True,
            )

        if any(k in ml_res for k in ("ld50", "half_life", "hepatic_clearance")):
            st.markdown("")
            pk_cols = st.columns(3)
            pk_items = [
                ("LD50 (oral)", ml_res.get("ld50"), " mg/kg"),
                ("Half-life", ml_res.get("half_life"), " hr"),
                ("Hepatic clearance", ml_res.get("hepatic_clearance"), " mL/min/g"),
            ]
            for col, (lbl, val, unit) in zip(pk_cols, pk_items):
                if val is None:
                    continue
                col.markdown(
                    f'<div class="admet-verdict-card" style="border-left:3px solid #556;">'
                    f'<div class="admet-verdict-label">{lbl}</div>'
                    f'<div class="admet-verdict-value">{val}{unit}</div></div>',
                    unsafe_allow_html=True,
                )

    np_, nb_ = len(props["pains_alerts"]), len(props["brenk_alerts"])
    st.markdown("")
    if np_ == 0 and nb_ == 0:
        st.success("No PAINS or BRENK structural alerts detected")
    else:
        if np_:
            st.warning(f"**PAINS — {np_} alert(s):** " + "  ·  ".join(sorted(set(props["pains_alerts"]))))
        if nb_:
            st.warning(f"**BRENK — {nb_} alert(s):** " + "  ·  ".join(sorted(set(props["brenk_alerts"]))))


def create_docking_session_files(workdir, session_name, receptor_result, ligand_results, dock_records, engine, redock_result=None):
    workdir = Path(workdir)
    ts = timestamp_tag()

    summary_rows = []
    for d in dock_records:
        summary_rows.append({
            "Compound": d["name"],
            "Best_Affinity_kcal_mol": round(d["top_score"], 3) if d["top_score"] is not None else "",
            "Status": "success" if d["top_score"] is not None else "failed",
        })
    summary_rows.sort(key=lambda r: (r["Status"] != "success", float(r["Best_Affinity_kcal_mol"]) if r["Best_Affinity_kcal_mol"] != "" else 999))
    summary_csv_path = str(workdir / "docking_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    report_path = str(workdir / "session_report.txt")
    with open(report_path, "w") as f:
        f.write(f"IMPPATez Docking Session Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Engine    : {engine}\n")
        f.write(f"Timestamp : {ts}\n\n")
        if redock_result and redock_result.get("success"):
            rmsd_v = redock_result.get("rmsd")
            f.write(f"Redocking Validation (RMSD, Pose 1)\n")
            f.write(f"  Ligand   : {redock_result.get('original_resname','')}\n")
            f.write(f"  RMSD     : {rmsd_v:.3f} Å\n" if rmsd_v is not None else "  RMSD     : N/A\n")
            f.write("\n")
        f.write(f"Docking Results (sorted by affinity)\n")
        f.write(f"{'-'*40}\n")
        for row in summary_rows:
            score_str = f"{row['Best_Affinity_kcal_mol']} kcal/mol" if row["Best_Affinity_kcal_mol"] != "" else "failed"
            f.write(f"  {row['Compound']:<35} {score_str}\n")


    zip_path = str(workdir / "docking_results.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_csv_path, arcname="docking_summary.csv")
        zf.write(report_path,      arcname="session_report.txt")

        for key, label in [
            ("receptor_pdb",   "rec.pdb"),
            ("receptor_pdbqt", "rec.pdbqt"),
            ("config_file",    "config.txt"),
            ("box_pdb",        "rec.box.pdb"),
        ]:
            fp = receptor_result.get(key)
            if fp and os.path.exists(fp):
                zf.write(fp, arcname=f"receptor/{label}")

        for lr in ligand_results:
            safe = safe_name(lr["name"])
            for key, ext in [("sdf", ".sdf"), ("pdbqt", ".pdbqt"), ("pdb", ".pdb")]:
                fp = lr.get(key)
                if fp and os.path.exists(fp):
                    zf.write(fp, arcname=f"ligands/{safe}{ext}")

        for d in dock_records:
            if d.get("top_score") is None:
                continue
            safe = safe_name(d["name"])
            if d.get("docked_sdf") and os.path.exists(d["docked_sdf"]):
                zf.write(d["docked_sdf"],   arcname=f"docked_poses/{safe}_docked.sdf")
            if d.get("docked_pdbqt") and os.path.exists(d["docked_pdbqt"]):
                zf.write(d["docked_pdbqt"], arcname=f"docked_poses/{safe}_docked.pdbqt")

        if redock_result and redock_result.get("success"):
            for key, label in [
                ("original_ligand_pdb", f"original_{redock_result.get('original_resname','LIG')}.pdb"),
                ("docked_sdf",          "redocked.sdf"),
                ("docked_pdbqt",        "redocked.pdbqt"),
            ]:
                fp = redock_result.get(key)
                if fp and os.path.exists(fp):
                    zf.write(fp, arcname=f"redocking/{label}")

    session_json = str(workdir / "session.json")
    session_data = {
        "session_name": session_name, "engine": engine, "timestamp": ts,
        "receptor": {
            "pdb": receptor_result.get("receptor_pdb"),
            "pdbqt": receptor_result.get("receptor_pdbqt"),
            "config": receptor_result.get("config_file"),
        },
        "docked_poses": [
            {"name": d["name"], "top_score": d.get("top_score"), "docked_sdf": d.get("docked_sdf")}
            for d in dock_records
        ],
    }
    with open(session_json, "w") as f:
        _json.dump(session_data, f, indent=2)

    return {"session_json": session_json, "links_txt": report_path, "zip": zip_path}


def fetch_page(plant):
    url = f"{BASE}/phytochemical/{urllib.parse.quote(plant)}"
    r = get_session().get(url, timeout=25)
    r.raise_for_status()
    return r.text

def fetch_detail_page(imphy_id):
    url = f"{BASE}/phytochemical-detailedpage/{imphy_id}"
    r = get_session().get(url, timeout=25)
    r.raise_for_status()
    return r.text

def extract_entries(html):
    entries = []
    try:
        for table in pd.read_html(html):
            cols = [str(c).strip() for c in table.columns]
            if "IMPPAT Phytochemical identifier" in cols and "Phytochemical name" in cols:
                sub = table[["IMPPAT Phytochemical identifier", "Phytochemical name"]].copy()
                sub.columns = ["IMPHY_ID", "Phytochemical_Name"]
                for _, row in sub.iterrows():
                    imphy_id = str(row["IMPHY_ID"]).strip()
                    name = str(row["Phytochemical_Name"]).strip()
                    if re.fullmatch(r"IMPHY\d+", imphy_id):
                        entries.append({"IMPHY_ID": imphy_id, "Phytochemical_Name": name})
                break
    except Exception:
        pass
    if entries:
        seen, deduped = set(), []
        for item in entries:
            if item["IMPHY_ID"] not in seen:
                seen.add(item["IMPHY_ID"]); deduped.append(item)
        return deduped
    pairs = re.findall(r"(IMPHY\d+)\s*<\/td>\s*<td[^>]*>\s*([^<]+)", html, flags=re.IGNORECASE)
    seen = set()
    for imphy_id, name in pairs:
        if imphy_id not in seen:
            seen.add(imphy_id)
            entries.append({"IMPHY_ID": imphy_id.strip(), "Phytochemical_Name": name.strip()})
    return entries

def extract_smiles(detail_html):
    text = re.sub(r"<[^>]+>", " ", detail_html)
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in [r"SMILES:\s*(.*?)\s*InChI:", r"SMILES:\s*(.*?)\s*InChIKey:", r"SMILES:\s*(.*?)\s*Functional groups:"]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m: return m.group(1).strip()
    return ""

def extract_cid(detail_html):
    for pattern in [r'https?://pubchem\.ncbi\.nlm\.nih\.gov/compound/(\d+)', r'https?://pubchem\.ncbi\.nlm\.nih\.gov/summary/summary\.cgi\?cid=(\d+)']:
        matches = re.findall(pattern, detail_html, re.IGNORECASE)
        if matches: return matches[0]
    text = re.sub(r"<[^>]+>", " ", detail_html)
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in [r"PubChem\s+CID[:\s]+(\d+)", r"CID[:\s]+(\d+)"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m: return m.group(1)
    return ""

def ro5_from_smiles(smiles):
    if not smiles:
        return "No SMILES"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return "Invalid SMILES"
        violations = sum([
            Descriptors.MolWt(mol)>500,
            Descriptors.MolLogP(mol)>5,
            Lipinski.NumHDonors(mol)>5,
            Lipinski.NumHAcceptors(mol)>10
        ])
        return "Passed" if violations <= 1 else f"Failed ({violations} violations)"
    except: return "RDKit Error"

def process_single_entry(item, max_retries=2):
    for attempt in range(max_retries+1):
        try:
            if attempt > 0: time.sleep(attempt)
            html = fetch_detail_page(item["IMPHY_ID"])
            smiles = extract_smiles(html)
            cid = extract_cid(html)
            return {
                "IMPHY_ID": item["IMPHY_ID"],
                "CID": cid,
                "Phytochemical_Name": item["Phytochemical_Name"],
                "SMILES_IMPPAT": smiles,
                "LIPINSKI": ro5_from_smiles(smiles) if smiles else "No SMILES"
            }
        except Exception:
            if attempt == max_retries:
                return {
                    "IMPHY_ID": item["IMPHY_ID"],
                    "CID": "",
                    "Phytochemical_Name": item["Phytochemical_Name"],
                    "SMILES_IMPPAT": "",
                    "LIPINSKI": "No SMILES"
                }
    return None

def get_bangla_name_from_wikipedia(plant_name):

    if not plant_name:
        return ""
    try:
        formatted = plant_name.strip().replace(" ", "_")

        wp_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(formatted)}"
            "&prop=pageprops&ppprop=wikibase_item&format=json"
        )
        r = requests.get(wp_url, timeout=10, headers=HEADERS)
        qid = None
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                qid = page.get("pageprops", {}).get("wikibase_item")
                if qid:
                    break

        if qid:
            wd_url = (
                f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
            )
            wd_r = requests.get(wd_url, timeout=10, headers=HEADERS)
            if wd_r.status_code == 200:
                entity = wd_r.json().get("entities", {}).get(qid, {})
                bn_label = entity.get("labels", {}).get("bn", {}).get("value", "")
                if bn_label:
                    return bn_label
                bn_aliases = entity.get("aliases", {}).get("bn", [])
                if bn_aliases:
                    return bn_aliases[0].get("value", "")

        bn_url = (
            f"https://bn.wikipedia.org/api/rest_v1/page/summary/"
            f"{urllib.parse.quote(formatted)}"
        )
        br = requests.get(bn_url, timeout=10, headers=HEADERS)
        if br.status_code == 200:
            data = br.json()
            desc = data.get("description", "")
            title = data.get("title", "")
            if title and any("\u0980" <= c <= "\u09FF" for c in title):
                return title
            if desc and any("\u0980" <= c <= "\u09FF" for c in desc):
                return desc

    except Exception:
        pass
    return ""

def extract_plant_info(html):
    info = {}
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    _stop = r"(?=Kingdom|Family|Group|Common\s+[Nn]ame|Synonym|System\s+of\s+medicine|More\s+Information|Total|$)"
    for key, pat in [
        ("Kingdom",
         r"Kingdom\s*:?\s*([A-Za-z][A-Za-z ]*?)" + _stop),
        ("Family",
         r"Family\s*:?\s*([A-Za-z][A-Za-z ]*?)" + _stop),
        ("Group",
         r"Group\s*:?\s*([A-Za-z][A-Za-z ]*?)" + _stop),
        ("Common name",
         r"Common\s+[Nn]ames?\s*:?\s*(.+?)" + _stop),
        ("Synonymous names",
         r"Synonymous\s+[Nn]ames?\s*:?\s*(.+?)"
         r"(?=System\s+of\s+medicine|More\s+Information|Kingdom|Family|Group|Common\s+[Nn]ame|Total|$)"),
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip(".")
            if val:
                info[key] = val
    return info

def run_step1(plant_name):
    plant_name = str(plant_name or "").strip()
    if not plant_name: return None, None, "Please enter a plant name."
    try:
        html = fetch_page(plant_name)
    except Exception as exc:
        return None, None, f"Failed to fetch: {exc}"
    plant_info = extract_plant_info(html)
    entries = extract_entries(html)
    if not entries: return None, None, "No phytochemicals found."
    rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_entry, item): item for item in entries}
        for i, future in enumerate(as_completed(futures)):
            status_text.text(f"Processing Compounds ({i+1}/{len(entries)})...")
            progress_bar.progress((i+1)/len(entries))
            try:
                r = future.result()
                if r: rows.append(r)
            except: pass
    progress_bar.empty(); status_text.empty()
    df = pd.DataFrame(rows)[["IMPHY_ID","CID","Phytochemical_Name","SMILES_IMPPAT","LIPINSKI"]]
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", plant_name).strip("_") or "plant"
    csv_path = unique_filepath(os.path.join(OUTPUT_DIR, f"{safe}_phytochemicals.csv"))
    df.to_csv(csv_path, index=False)
    total = len(df); passed = int((df["LIPINSKI"]=="Passed").sum()); failed = total - passed
    return df, csv_path, plant_info, total, passed, failed


def download_sdf(imphy_id, folder):
    url = f"{BASE}/images/3D/SDF/{imphy_id}_3D.sdf"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=25, headers=HEADERS)
            if r.status_code == 200 and len(r.text.strip()) > 50:
                path = os.path.join(folder, f"{imphy_id}.sdf")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                return path
        except: pass
        if attempt < 2: time.sleep(1+attempt)
    return None

def run_step2(csv_path, sdf_mode, show_progress=True):
    if not csv_path or not os.path.exists(csv_path):
        return None, None, "No CSV file found. Run Step 1 first."
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        return None, None, f"Could not read CSV: {exc}"
    if "IMPHY_ID" not in df.columns:
        return None, None, "CSV must contain IMPHY_ID column."
    work_df = df.copy()
    if sdf_mode in ("RO5 Passed", "Lipinski Passed Only"):
        lipinski = work_df["LIPINSKI"] if "LIPINSKI" in work_df.columns else pd.Series("", index=work_df.index)
        work_df = work_df[lipinski.fillna("").str.strip()=="Passed"]
    folder = str(unique_workdir(Path(OUTPUT_DIR) / "sdf_files"))
    os.makedirs(folder, exist_ok=True)
    saved, missing = [], []
    ids = work_df["IMPHY_ID"].dropna().astype(str).str.strip().tolist()
    ids = [x for x in ids if x]
    progress_bar = st.progress(0) if show_progress else None
    status_text = st.empty() if show_progress else None
    for i, imphy in enumerate(ids):
        if show_progress:
            status_text.text(f"Extracting SDF ({i+1}/{len(ids)})...")
            progress_bar.progress((i+1)/len(ids))
        if re.fullmatch(r"IMPHY\d+", imphy):
            path = download_sdf(imphy, folder)
            if path: saved.append(path)
            else: missing.append(imphy)
        else: missing.append(imphy)
    if show_progress:
        progress_bar.empty(); status_text.empty()
    zip_name = None
    if saved:
        zip_name = unique_filepath(os.path.join(OUTPUT_DIR, "IMPPAT_3D_SDF.zip"))
        with zipfile.ZipFile(zip_name, "w") as z:
            for fp in saved: z.write(fp, arcname=os.path.basename(fp))
    return zip_name, missing, len(saved), len(missing)


HYDROPHOBIC_RESNAMES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS", "TYR"}
POSITIVE_RESNAMES = {"LYS", "ARG", "HIS"}
NEGATIVE_RESNAMES = {"ASP", "GLU"}
CHARGED_RESNAMES = POSITIVE_RESNAMES | NEGATIVE_RESNAMES
POLAR_ATOM_ELEMENTS = {"N", "O"}


def _parse_receptor_atoms_for_interactions(receptor_pdb_path):
    """Lightweight fixed-column PDB parser -> list of atom dicts used for contact geometry."""
    atoms = []
    if not receptor_pdb_path or not os.path.exists(receptor_pdb_path):
        return atoms
    with open(receptor_pdb_path) as f:
        for line in f:
            field = line[:6].strip()
            if field not in ("ATOM", "HETATM"):
                continue
            try:
                name = line[12:16].strip()
                resname = line[17:20].strip().upper()
                chain = line[21].strip() if len(line) > 21 and line[21].strip() else "A"
                resid = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                element = (line[76:78].strip() if len(line) >= 78 else "").upper()
                if not element:
                    element = re.sub(r"[^A-Za-z]", "", name)[:1].upper()
                if element == "H" or resname in EXCLUDE_IONS:
                    continue
                atoms.append({
                    "name": name, "resname": resname, "chain": chain,
                    "resid": resid, "coord": (x, y, z), "element": element,
                })
            except Exception:
                continue
    return atoms


def _get_best_pose_ligand_mol(docked_sdf, docked_pdbqt, engine):
    """Return an RDKit Mol (3D, top-ranked pose) regardless of docking engine used."""
    sdf_path = docked_sdf
    if (not sdf_path or not os.path.exists(sdf_path)) and docked_pdbqt and os.path.exists(docked_pdbqt):
        try:
            sdf_path = convert_pdbqt_to_sdf(docked_pdbqt)
        except Exception:
            sdf_path = None
    if not sdf_path or not os.path.exists(sdf_path):
        return None
    try:
        supp = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        for mol in supp:
            if mol is not None and mol.GetNumConformers() > 0:
                return mol
    except Exception:
        pass
    return None


def analyze_binding_interactions(receptor_pdb_path, ligand_mol,
                                  hbond_cutoff=3.5, hydrophobic_cutoff=4.5, electrostatic_cutoff=4.5):
    """
    Simplified, distance-based interaction fingerprint between a docked ligand pose and the
    receptor — approximates the residue classes reported by tools like LigPlot+/PLIP:
      • Electrostatic / salt-bridge : ligand N/O <-> charged-residue side-chain N/O (<= electrostatic_cutoff)
      • Hydrogen bond               : ligand N/O <-> receptor N/O                  (<= hbond_cutoff)
      • Hydrophobic contact         : ligand C   <-> hydrophobic-residue side-chain C (<= hydrophobic_cutoff)
    Geometric heuristic only (no H-bond angle / formal-charge perception) — meant for fast
    triage across many docked compounds, not a publication-grade PLIP/LigPlot+ run.
    """
    result = {
        "hbond_residues": [], "hydrophobic_residues": [], "electrostatic_residues": [],
    }
    if ligand_mol is None:
        return result
    rec_atoms = _parse_receptor_atoms_for_interactions(receptor_pdb_path)
    if not rec_atoms:
        return result

    conf = ligand_mol.GetConformer()
    lig_atoms = [
        {"element": a.GetSymbol(), "coord": tuple(conf.GetAtomPosition(a.GetIdx()))}
        for a in ligand_mol.GetAtoms() if a.GetSymbol() != "H"
    ]
    if not lig_atoms:
        return result

    lig_coords = np.array([a["coord"] for a in lig_atoms])
    rec_coords = np.array([a["coord"] for a in rec_atoms])
    dmat = distance_matrix(lig_coords, rec_coords)

    hbond_res, hydrophobic_res, electro_res = {}, {}, {}

    def _register(bucket, ra, d):
        tag = f"{ra['resname']}{ra['resid']}:{ra['chain']}"
        entry = bucket.get(tag)
        if entry is None or d < entry["min_dist"]:
            bucket[tag] = {"resname": ra["resname"], "resid": ra["resid"], "chain": ra["chain"], "min_dist": d}

    for li, la in enumerate(lig_atoms):
        la_polar = la["element"] in POLAR_ATOM_ELEMENTS
        for ri, ra in enumerate(rec_atoms):
            d = dmat[li, ri]
            ra_polar = ra["element"] in POLAR_ATOM_ELEMENTS

            if (la_polar and ra_polar and ra["resname"] in CHARGED_RESNAMES
                    and ra["name"] not in _BACKBONE and d <= electrostatic_cutoff):
                _register(electro_res, ra, d)
            elif la_polar and ra_polar and d <= hbond_cutoff:
                _register(hbond_res, ra, d)
            elif (la["element"] == "C" and ra["element"] == "C" and ra["resname"] in HYDROPHOBIC_RESNAMES
                    and ra["name"] not in _BACKBONE and d <= hydrophobic_cutoff):
                _register(hydrophobic_res, ra, d)

    result["hbond_residues"] = sorted(hbond_res.values(), key=lambda r: r["min_dist"])
    result["hydrophobic_residues"] = sorted(hydrophobic_res.values(), key=lambda r: r["min_dist"])
    result["electrostatic_residues"] = sorted(electro_res.values(), key=lambda r: r["min_dist"])
    return result


def _score_interaction_profile(interactions):
    """Weighted contact-richness score, used only to help rank compounds (not a physical energy)."""
    return (2.0 * len(interactions.get("hbond_residues", []))
            + 1.5 * len(interactions.get("electrostatic_residues", []))
            + 1.0 * len(interactions.get("hydrophobic_residues", [])))


def rank_preferred_compounds(rows):
    """
    rows: list of dicts with 'score' (binding energy, more negative = better) and
    'interaction_score' (higher = better). Returns rows sorted by a composite of
    normalised docking score (55%) and normalised interaction richness (45%).
    """
    if not rows:
        return rows
    scores = [r["score"] for r in rows]
    smin, smax = min(scores), max(scores)
    iscores = [r["interaction_score"] for r in rows]
    imin, imax = min(iscores), max(iscores)
    for r in rows:
        score_norm = 1.0 if smax == smin else (smax - r["score"]) / (smax - smin)
        inter_norm = 1.0 if imax == imin else (r["interaction_score"] - imin) / (imax - imin)
        r["composite"] = 0.55 * score_norm + 0.45 * inter_norm
    return sorted(rows, key=lambda r: r["composite"], reverse=True)


def render_ligplot_diagram(ligand_mol, interactions, compound_name, score):
    """Simplified LigPlot-style 2D diagram: 2D ligand structure in the centre, with
    interacting residues arranged around it and colour-coded, dashed contact lines."""
    mol2d = Chem.Mol(ligand_mol)
    try:
        mol2d = Chem.RemoveHs(mol2d)
    except Exception:
        pass
    try:
        AllChem.Compute2DCoords(mol2d)
    except Exception:
        pass
    img = Draw.MolToImage(mol2d, size=(420, 420))

    color_map = {"hbond": "#1b7a3d", "hydrophobic": "#b34700", "electrostatic": "#1155cc"}
    label_map = {"hbond": "H-bond", "hydrophobic": "Hydrophobic", "electrostatic": "Electrostatic"}

    residues = []
    for kind, key in (("hbond", "hbond_residues"), ("electrostatic", "electrostatic_residues"),
                       ("hydrophobic", "hydrophobic_residues")):
        for r in interactions.get(key, []):
            residues.append((f"{r['resname']}{r['resid']}", kind, r["min_dist"]))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.imshow(img, extent=(-0.55, 0.55, -0.55, 0.55), zorder=2)
    ax.set_title(f"{compound_name}  —  {score:.2f} kcal/mol", fontsize=12, fontweight="bold", pad=10)

    n = max(len(residues), 1)
    for i, (label, kind, dist) in enumerate(residues):
        angle = 2 * np.pi * i / n
        ix, iy = 0.55 * np.cos(angle), 0.55 * np.sin(angle)
        rx, ry = 1.2 * np.cos(angle), 1.2 * np.sin(angle)
        ax.plot([ix, rx], [iy, ry], linestyle="--", linewidth=1.6, color=color_map[kind], zorder=1)
        ax.text(rx, ry, f"{label}\n{dist:.2f} Å", ha="center", va="center", fontsize=8,
                color=color_map[kind], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color_map[kind], lw=1.2), zorder=4)

    if residues:
        legend_handles = [plt.Line2D([0], [0], color=c, lw=2, linestyle="--", label=label_map[k])
                           for k, c in color_map.items()]
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
                  ncol=3, frameon=False, fontsize=9)
    else:
        ax.text(0, -1.0, "No residue contacts detected within cutoff", ha="center", va="center",
                fontsize=9, color="#888")

    fig.tight_layout()
    return fig



def plot_score_chart(dock_records, engine, ref_score=None, ref_label="Reference"):
    df = pd.DataFrame([{
        "name": d["name"],
        "score": d.get("top_score") or 0
    } for d in dock_records if d.get("top_score") is not None]).sort_values("score", ascending=True).reset_index(drop=True)
    if df.empty:
        return None
    width = max(7, 0.7*len(df)+2)
    fig, ax = plt.subplots(figsize=(width, 5.0))
    ax.scatter(range(len(df)), df["score"], s=80, color="steelblue", zorder=3, label="Docked compounds")
    for i, v in enumerate(df["score"]):
        ax.text(i, v+0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color="dimgray")
    handles = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor="steelblue",markersize=8,label="Docked compounds")]
    if ref_score is not None:
        ax.axhline(ref_score, color="crimson", linewidth=1.5, linestyle="--", zorder=2)
        handles.append(plt.Line2D([0],[0],color="crimson",linewidth=1.5,linestyle="--",label=f"{ref_label} ({ref_score:.1f} kcal/mol)"))
    ax.legend(handles=handles, fontsize=9, framealpha=0.9, loc="lower left", bbox_to_anchor=(0,1.12), ncol=len(handles), borderaxespad=0)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["name"], rotation=55, ha="right", fontsize=9)
    ax.set_xlabel("Compound", fontsize=11)
    ax.set_ylabel("Binding energy (kcal/mol)", fontsize=11)
    ax.set_title(f"Top-pose {engine} score — best pose per compound", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.subplots_adjust(top=0.82)
    return fig


for key, default in [
    ("csv_path", None), ("df", None),
    ("receptor_result", None), ("receptor_pdb_path", None), ("cocrystal_info", None),
    ("dock_records", []), ("ligand_results", []), ("redock_result", None),
    ("session_files", None), ("dock_bin_path", None), ("current_engine", None),
    ("active_step", 1), ("interaction_top10_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

render_pipeline_stepper()

if st.session_state.active_step == 1:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">Step 1 · Phytochemical Library</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("imppat_form"):

        col1, col2 = st.columns([5, 1])

        with col1:
            plant_name = st.text_input(
                "Scientific Name:",
                placeholder="e.g., Azadirachta indica",
                key="plant_input",
                label_visibility="visible"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.form_submit_button(
                "🔍 Search IMPPAT",
                use_container_width=True
            )

    if search_btn and plant_name:
        with st.spinner(f"Searching IMPPAT for {plant_name}..."):
            result = run_step1(plant_name)
        if result[0] is not None:
            df, csv_path, plant_info, total, passed, failed = result
            st.session_state.df = df
            st.session_state.csv_path = csv_path
            st.session_state["phyto_total"] = total
            st.session_state["phyto_passed"] = passed
            st.session_state["phyto_failed"] = failed
            st.session_state["phyto_plant_info"] = plant_info
            st.session_state["phyto_plant_name"] = plant_name
            st.session_state["phyto_bangla"] = get_bangla_name_from_wikipedia(plant_name)
            st.session_state["phyto_chart_filter"] = None
        else:
            st.error(result[2] if len(result)>2 else "No phytochemicals found.")

    if st.session_state.get("df") is not None and st.session_state.get("phyto_total") is not None:
        df       = st.session_state.df
        csv_path = st.session_state.csv_path
        total    = st.session_state["phyto_total"]
        passed   = st.session_state["phyto_passed"]
        failed   = st.session_state["phyto_failed"]
        plant_info  = st.session_state["phyto_plant_info"]
        _plant_name = st.session_state["phyto_plant_name"]
        _bangla     = st.session_state.get("phyto_bangla", "")

        _parts    = _plant_name.strip().split()
        _fmt_name = (_parts[0].capitalize() + " " + " ".join(p.lower() for p in _parts[1:])) if len(_parts) >= 2 else _plant_name.strip().capitalize()
        _kingdom  = plant_info.get("Kingdom", "") or "—"
        _family   = plant_info.get("Family", "") or "—"
        _group    = plant_info.get("Group", "") or "—"
        _common   = plant_info.get("Common name", "") or "—"
        _synonyms = plant_info.get("Synonymous names", "") or "—"
        _bangla_d = _bangla or "—"

        _card_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:transparent; font-family:'IBM Plex Sans',sans-serif; padding:2px 0; }}
.card {{
  background:linear-gradient(135deg,#f0fbf0 0%,#e4f5e5 100%);
  border:1.5px solid #a5d6a7; border-radius:16px;
  padding:18px 22px 16px 22px; box-shadow:0 4px 20px rgba(46,125,50,0.12);
}}
.header {{ display:flex; align-items:center; gap:12px; margin-bottom:14px; }}
.icon {{ background:linear-gradient(135deg,#2e7d32,#43a047); border-radius:10px; padding:8px 12px; font-size:1.5rem; line-height:1; }}
.plant-name {{ font-size:1.15rem; font-weight:800; color:#1a3d20; font-style:italic; }}
.sub-label {{ font-size:0.74rem; color:#4a7a52; font-weight:500; margin-top:2px; }}
.info-table {{ background:rgba(255,255,255,0.6); border-radius:10px; padding:6px 14px; }}
.row {{ display:flex; align-items:baseline; gap:10px; padding:5px 0; border-bottom:1px solid rgba(46,125,50,0.1); }}
.row:last-child {{ border-bottom:none; }}
.key {{ font-size:0.75rem; color:#4a7a52; font-weight:700; min-width:155px; flex-shrink:0; }}
.val {{ font-size:0.86rem; color:#1f3d25; }}
</style></head><body>
<div class="card">
  <div class="header">
    <div class="icon">🌿</div>
    <div>
      <div class="plant-name">{_fmt_name}</div>
      <div class="sub-label">IMPPATez Phytochemical Profile</div>
    </div>
  </div>
  <div class="info-table">
    <div class="row"><span class="key">✰ Kingdom</span><span class="val">{_kingdom}</span></div>
    <div class="row"><span class="key">✰ Family</span><span class="val">{_family}</span></div>
    <div class="row"><span class="key">✰ Group</span><span class="val">{_group}</span></div>
    <div class="row"><span class="key">✰ Common name</span><span class="val">{_common}</span></div>
    <div class="row">
        <span class="key">✰ Synonymous names</span>
        <span class="val" style="font-style: italic;">{_synonyms}</span>
    </div>
    <div class="row"><span class="key">✰ বাংলা নাম</span><span class="val">{_bangla_d}</span></div>
  </div>
</div>
</body></html>"""
        components.html(_card_html, height=300, scrolling=False)

        _sc1, _sc2, _sc3 = st.columns(3)
        with _sc1:
            st.markdown(f"""<div style="background:#e8f5e9;color:#1b5e20;border-radius:12px;
                padding:14px;text-align:center;margin-bottom:6px;">
                <div style="font-size:1.6rem;font-weight:800;line-height:1.1;">{total}</div>
                <div style="font-size:0.70rem;font-weight:600;text-transform:uppercase;margin-top:4px;">Total Phytochemicals</div>
            </div>""", unsafe_allow_html=True)
            if st.button("☘️ View All", key="btn_chart_all", use_container_width=True):
                st.session_state["phyto_chart_filter"] = "all"
        with _sc2:
            st.markdown(f"""<div style="background:#d4edda;color:#155724;border-radius:12px;
                padding:14px;text-align:center;margin-bottom:6px;">
                <div style="font-size:1.6rem;font-weight:800;line-height:1.1;">{passed}</div>
                <div style="font-size:0.70rem;font-weight:600;text-transform:uppercase;margin-top:4px;">RO5 Passed</div>
            </div>""", unsafe_allow_html=True)
            if st.button("✅ View RO5 Passed", key="btn_chart_passed", use_container_width=True):
                st.session_state["phyto_chart_filter"] = "passed"
        with _sc3:
            st.markdown(f"""<div style="background:#fdecea;color:#7f1d1d;border-radius:12px;
                padding:14px;text-align:center;margin-bottom:6px;">
                <div style="font-size:1.6rem;font-weight:800;line-height:1.1;">{failed}</div>
                <div style="font-size:0.70rem;font-weight:600;text-transform:uppercase;margin-top:4px;">RO5 Failed</div>
            </div>""", unsafe_allow_html=True)
            if st.button("🚫 View RO5 failed", key="btn_chart_failed", use_container_width=True):
                st.session_state["phyto_chart_filter"] = "failed"

        _chart_filter = st.session_state.get("phyto_chart_filter", None)
        if _chart_filter is not None:
            _plot_df = df.copy()
            if _chart_filter == "passed":
                _plot_df = _plot_df[_plot_df["LIPINSKI"] == "Passed"]
                _table_title = f"✅ RO5 Passed Compounds : {len(_plot_df)} compounds"
            elif _chart_filter == "failed":
                _plot_df = _plot_df[_plot_df["LIPINSKI"] != "Passed"]
                _table_title = f"❌ RO5 Failed Compounds : {len(_plot_df)} compounds"
            else:
                _table_title = f"🌿 All Phytochemicals : {len(_plot_df)} compounds"

            st.markdown(f"**{_table_title}**")
            if len(_plot_df) > 0:
                st.dataframe(_plot_df, use_container_width=True, height=350, hide_index=True)
            else:
                st.info("No compounds to display for this filter.")

        with open(csv_path, "rb") as f:
            st.download_button("Export CSV", data=f,
                               file_name=os.path.basename(csv_path), mime="text/csv",
                               use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Step 2 · 3D Structures</div>
        <div class="step-heading">Download SDF files for selected compounds</div>
    </div>
    """, unsafe_allow_html=True)

    sdf_mode = st.radio("Download Filter", ["All Phytochemicals","RO5 Passed"], horizontal=True)
    if st.button("Download ZIP", use_container_width=True, type="primary"):
        if st.session_state.csv_path:
            with st.spinner("Preparing your SDF download…"):
                zip_path, missing_ids, saved_count, missing_count = run_step2(
                    st.session_state.csv_path, sdf_mode, show_progress=False
                )
            st.session_state["sdf_zip_path"]        = zip_path
            st.session_state["sdf_missing_ids"]     = missing_ids or []
            st.session_state["sdf_saved_count"]     = saved_count
            st.session_state["sdf_missing_count"]   = missing_count
            st.session_state["sdf_auto_dl_pending"] = True
        else:
            st.warning("Run Step 1 first.")

    if st.session_state.get("sdf_zip_path"):
        _zip      = st.session_state["sdf_zip_path"]
        _saved    = st.session_state["sdf_saved_count"]
        _miss_ids = st.session_state["sdf_missing_ids"]
        _miss_n   = st.session_state["sdf_missing_count"]

        if st.session_state.get("sdf_auto_dl_pending") and _zip and os.path.exists(_zip):
            trigger_browser_download(_zip)
            st.session_state["sdf_auto_dl_pending"] = False

        if _miss_n > 0 and _miss_ids:
            _links_html = " &nbsp;|&nbsp; ".join(
                f'<a href="https://cb.imsc.res.in/imppat/phytochemical-detailedpage/{mid}" '
                f'target="_blank" style="color:#c0392b;font-weight:600;text-decoration:underline;">'
                f'{mid}</a>'
                for mid in _miss_ids
            )
        else:
            _links_html = "None"

        st.markdown(
            f'<div style="background:#fdecea;border:1px solid #f5c6cb;border-radius:8px;'
            f'padding:8px 14px;font-size:0.88rem;color:#7f1d1d;margin-bottom:12px;">'
            f'⚠️ <b>Missing {_miss_n} SDF file(s):</b> {_links_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

        if os.path.exists(_zip):
            with open(_zip, "rb") as f:
                st.download_button(
                    "⬇️ Download Manually",
                    data=f,
                    file_name=os.path.basename(_zip),
                    mime="application/zip",
                    use_container_width=True,
                )

        st.caption("*Use Manual Option Only If Blocked by Your Browser.*")
 
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.active_step == 2:

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Step 3 · Docking Engine</div>
        <div class="step-heading">Choose engine and set parameters</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Engine & Binary")

    col_eng1, col_eng2 = st.columns([1, 1])

    with col_eng1:
        engine = st.selectbox(
            "Docking engine",
            ["VINA", "VINAXB", "GNINA", "UNIDOCK"],
            format_func=lambda x: {
                "VINA": "⚡ AutoDock Vina",
                "VINAXB": "🧲 VinaXB",
                "GNINA": "🧠 GNINA",
                "UNIDOCK": "🚀 Uni-Dock"
            }[x],
            key="main_engine_select"
        )
        st.session_state.current_engine = engine

    with col_eng2:
        custom_bin = st.text_input(
            "Custom binary path",
            key="custom_bin_input",
            help="Find binary: **macOS/Linux**: which <engine> **Windows**: where <engine>"
        )

    if st.button("Find Binary", key="find_binary_btn", use_container_width=True):

        with st.spinner(f"Looking for {engine} binary..."):

            dock_bin, source = find_docking_binary(engine, custom_bin)

            if dock_bin:

                st.session_state.dock_bin_path = dock_bin

                if engine == "GNINA" and _is_docker_gnina(dock_bin):
                    st.success("✅ GNINA will run through Docker image `gnina/gnina`")
                    st.caption("First run can take a while because Docker may need to download the image.")

                else:

                    if source == "custom":
                        st.caption("✅ Using Custom Binary Path")
                    else:
                        st.caption("✅ Using Auto-Detected Binary")

            else:

                st.session_state.dock_bin_path = None
                st.error(f"❌ {engine} Binary Not Found!")

    elif engine == "GNINA":

        ok, msg = docker_is_ready()

        if ok:
            st.info("🍎 macOS GNINA option: click **Find Binary** to use Docker image `gnina/gnina`.")
        else:
            st.warning(f"GNINA on macOS needs Docker or a Linux GNINA binary. Docker status: {msg}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Docking Parameters")

    if engine in ("VINA", "VINAXB"):
        p_row1a, p_row1b = st.columns(2)
        with p_row1a:
            exhaustiveness = st.slider("Exhaustiveness", 4, 128, 32, 2, key="exhaust")
        with p_row1b:
            num_modes = st.slider("Number of binding modes", 1, 40, 20, 1, key="modes")

        p_row2a, p_row2b = st.columns(2)
        with p_row2a:
            energy_range = st.slider("Energy range (kcal/mol)", 1, 10, 5, 1, key="energy")
        with p_row2b:
            ph_val = st.number_input("Target pH for protonation", 0.0, 14.0, 7.4, 0.1, key="ph_val")

        dock_params = {"exhaustiveness": exhaustiveness, "num_modes": num_modes, "energy_range": energy_range}

    elif engine == "GNINA":
        gnina_preset = st.selectbox(
            "GNINA protocol",
            ["Quick test", "Balanced", "CNN rescore"],
            help="Quick test is best on Mac/Docker for checking the pipeline. Balanced/CNN rescore are slower.",
            key="gnina_preset"
        )
        default_exhaust, default_modes, default_cnn = {
            "Quick test": (4, 1, "none"),
            "Balanced": (16, 9, "none"),
            "CNN rescore": (16, 20, "rescore"),
        }[gnina_preset]

        p_row1a, p_row1b = st.columns(2)
        with p_row1a:
            gnina_exhaust = st.slider("Exhaustiveness", 1, 128, default_exhaust, 1, key="gnina_exhaust")
        with p_row1b:
            gnina_modes = st.slider("Number of binding modes", 1, 40, default_modes, 1, key="gnina_modes")

        p_row2a, p_row2b = st.columns(2)
        with p_row2a:
            gnina_cnn = st.selectbox("CNN scoring mode", ["none", "rescore"], index=0 if default_cnn == "none" else 1, key="gnina_cnn")
        with p_row2b:
            ph_val = st.number_input("Target pH for protonation", 0.0, 14.0, 7.4, 0.1, key="ph_val")

        dock_params = {"gnina_exhaustiveness": gnina_exhaust, "gnina_num_modes": gnina_modes, "gnina_cnn_scoring": gnina_cnn}

    else:
        p_row1a, p_row1b = st.columns(2)
        with p_row1a:
            ud_search = st.selectbox("Search mode", ["fast", "balance", "detail", "custom"], key="ud_search")
        with p_row1b:
            ud_modes = st.slider("Number of binding modes", 1, 40, 20, 1, key="ud_modes")

        p_row2a, p_row2b = st.columns(2)
        with p_row2a:
            ud_scoring = st.selectbox("Scoring function", ["vina", "vinardo"], key="ud_scoring")
        with p_row2b:
            ph_val = st.number_input("Target pH for protonation", 0.0, 14.0, 7.4, 0.1, key="ph_val")

        dock_params = {"unidock_search_mode": ud_search, "unidock_num_modes": ud_modes, "unidock_scoring": ud_scoring}

    st.session_state.dock_params = dock_params

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Proceed", key="engine_proceed_btn", type="primary", use_container_width=True):
        st.session_state.active_step = 3
        st.rerun()

if st.session_state.active_step == 3:
    if not PRODY_AVAILABLE:
        st.warning("⚠️ ProDy not installed. Install with `pip install prody` for co-crystal ligand auto-detection and RMSD calculation.")
    if not check_obabel():
        st.error("❌ OpenBabel (obabel) not found. Install with `conda install -c conda-forge openbabel`")

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Step 4 · Receptor</div>
        <div class="step-heading">Fetch & Prepare Target Protein</div>
    """, unsafe_allow_html=True)

    rec_source = st.selectbox(
        " ",
        ["Search by PDB ID", "Search by Protein Name", "Upload PDB File"],
        key="receptor_source_select"
    )

    receptor_pdb_path = None

    if rec_source == "Search by PDB ID":
        st.markdown("PDB ID")
        col1, col2 = st.columns([3, 1])
        with col1:
            pdb_id_input = st.text_input(
                "PDB ID", placeholder="Enter 4-character PDB ID (e.g. 4EY7)",
                label_visibility="collapsed"
            )
        with col2:
            direct_download = st.button("Direct Download", use_container_width=True)
        
        if direct_download and pdb_id_input:
            pdb_id = pdb_id_input.strip().upper()
            if re.match(r'^[0-9A-Z]{4}$', pdb_id):
                with st.spinner(f"Downloading PDB structure {pdb_id} from RCSB..."):
                    out = os.path.join(OUTPUT_DIR, f"{pdb_id}.pdb")
                    if download_pdb_direct(pdb_id, out):
                        receptor_pdb_path = out
                        st.session_state.receptor_pdb_path = out
                        
                        info = get_pdb_info(pdb_id)
                        if info.get("title"):
                                st.info(f"📄 PDB Title: {info['title'][:1000]}")
                    else:
                        st.error(f"Failed to download {pdb_id}")
            else:
                st.error("Invalid PDB ID format")

    elif rec_source == "Search by protein name":
        search_q = st.text_input("Search by protein name or gene", 
                                 placeholder="e.g., EGFR, COVID-19 main protease, BRD4")
        
        if search_q:
            with st.spinner(f"Searching RCSB for '{search_q}'..."):
                hits = search_rcsb_pdb(search_q, top_n=15)
            
            if hits:
                
                result_options = []
                for h in hits:
                    display = f"{h['pdb_id']}"
                    if h.get('resolution') and h['resolution'] != 'N/A':
                        try:
                            res_val = float(h['resolution']) if isinstance(h['resolution'], (int, float)) else float(h['resolution'])
                            display += f" | {res_val:.1f}Å"
                        except:
                            display += f" | {h['resolution']}"
                    if h.get('method') and h['method'] != 'N/A':
                        display += f" | {h['method']}"
                    if h.get('title'):
                        display += f" | {h['title']}"
                    result_options.append((h['pdb_id'], display))
                
                selected_pdb = st.selectbox("Select structure to download",
                                            options=[r[0] for r in result_options],
                                            format_func=lambda x: next((r[1] for r in result_options if r[0]==x), x))
                
                if st.button("Download Structure", type="secondary", use_container_width=True):
                    out = os.path.join(OUTPUT_DIR, f"{selected_pdb}.pdb")
                    with st.spinner(f"Downloading {selected_pdb}..."):
                        if download_pdb_direct(selected_pdb, out):
                            receptor_pdb_path = out
                            st.session_state.receptor_pdb_path = out
                        else:
                            st.error(f"Failed to download {selected_pdb}")

    elif rec_source == "Upload PDB file":
        uploaded = st.file_uploader("Upload PDB file", type=["pdb","cif","ent"])
        if uploaded:
            out = unique_filepath(os.path.join(OUTPUT_DIR, "receptor_uploaded.pdb"))
            with open(out, "wb") as f: 
                f.write(uploaded.getvalue())
            receptor_pdb_path = out
            st.session_state.receptor_pdb_path = out
            st.success(f"✅ Uploaded: {uploaded.name}")

    if st.session_state.receptor_pdb_path and os.path.exists(st.session_state.receptor_pdb_path):
        receptor_pdb_path = st.session_state.receptor_pdb_path
        st.markdown(
            f"""
            <div class="current-receptor-card">
                <div class="current-receptor-label">Current receptor</div>
                <div class="current-receptor-path">📄 {receptor_pdb_path}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        file_size = os.path.getsize(receptor_pdb_path) / 1024
        st.caption(f"File size: {file_size:.1f} KB")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Step 5 · Grid Box</div>
        <div class="step-heading">Define the docking search space</div>
    """, unsafe_allow_html=True)

    if receptor_pdb_path and os.path.exists(receptor_pdb_path):
        if st.button("Detect Co-crystal Ligand", type="secondary"):
            if PRODY_AVAILABLE:
                with st.spinner("Analyzing PDB file for co-crystal ligands..."):
                    info = detect_cocrystal_ligand(receptor_pdb_path)
                if info["found"]:
                    st.session_state.cocrystal_info = info
                else:
                    st.warning(f"⚠️ {info.get('error', 'No ligand detected')}")
            else:
                st.error("ProDy not installed")

    if st.session_state.get("cocrystal_info") and st.session_state.cocrystal_info.get("found"):
        info = st.session_state.cocrystal_info
        chain = info.get("chain") or "-"
        center = info.get("center") or (0.0, 0.0, 0.0)
        st.markdown(
            f"""
            <div class="ligand-detect-card">
                <div class="ligand-detect-item">
                    <div class="ligand-detect-label">Detected ligand</div>
                    <div class="ligand-detect-value">✅ {info.get('resname', 'LIG')} (Chain {chain})</div>
                </div>
                <div class="ligand-detect-item">
                    <div class="ligand-detect-label">Center Coordinates</div>
                    <div class="ligand-detect-value">📍 ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    center_mode = st.radio("Box Center",
                           ["Use co-crystal ligand center", "Enter XYZ manually"],
                           horizontal=True)

    auto_cx = auto_cy = auto_cz = None
    if center_mode == "Use co-crystal ligand center" and st.session_state.cocrystal_info:
        cx, cy, cz = st.session_state.cocrystal_info["center"]
        auto_cx, auto_cy, auto_cz = cx, cy, cz
        st.info(f"📍 Using co-crystal center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: auto_cx = st.number_input("Center X (Å)", value=0.0, step=5.0, format="%.3f")
        with c2: auto_cy = st.number_input("Center Y (Å)", value=0.0, step=5.0, format="%.3f")
        with c3: auto_cz = st.number_input("Center Z (Å)", value=0.0, step=5.0, format="%.3f")

    st.markdown("#### Box Size (Å)")
    bs1, bs2, bs3 = st.columns(3)
    with bs1: box_sx = st.slider("Size X (Å)", 10, 40, 20, 2, key="box_sx")
    with bs2: box_sy = st.slider("Size Y (Å)", 10, 40, 20, 2, key="box_sy")
    with bs3: box_sz = st.slider("Size Z (Å)", 10, 40, 20, 2, key="box_sz")
    
    volume = box_sx * box_sy * box_sz
    st.info(f"📐 Box volume: **{volume:,} Å³**")

    st.markdown("#### Chain & Cofactor Selection")
    keep_chain = "All chains"
    keep_cofactor_keys = []
    if receptor_pdb_path and os.path.exists(receptor_pdb_path):
        cc_info = detect_chains_and_cofactors(receptor_pdb_path)
        if cc_info["chains"]:
            keep_chain = st.selectbox(
                "Which chain do you want to keep?",
                ["All chains"] + cc_info["chains"],
                key="keep_chain_select"
            )
        else:
            st.caption("No chain information detected in this PDB.")

        if cc_info["cofactors"]:
            cof_options = [
                f"{c['resname']} (chain {c['chain'] or '-'}, resid {c['resid']})"
                for c in cc_info["cofactors"]
            ]
            cof_lookup = {
                lbl: (c["resname"], c["chain"], c["resid"])
                for lbl, c in zip(cof_options, cc_info["cofactors"])
            }
            selected_cofs = st.multiselect(
                "Which cofactor(s) do you want to keep? (unselected ones are stripped)",
                cof_options,
                key="keep_cofactors_select"
            )
            keep_cofactor_keys = [cof_lookup[l] for l in selected_cofs]
        else:
            st.caption("No cofactor-like residues (e.g. HEM, ATP, FAD, buffer components) detected.")
    else:
        st.caption("Load a receptor above to choose a chain and cofactors to keep.")

    if st.button("Prepare Receptor", type="primary", use_container_width=True):
        if not receptor_pdb_path or not os.path.exists(receptor_pdb_path):
            st.error("❌ Please load a receptor PDB file first.")
        elif not check_obabel():
            st.error("❌ OpenBabel not found.")
        else:
            work = unique_workdir(Path(OUTPUT_DIR) / "receptor_prep")
            work.mkdir(parents=True, exist_ok=True)
            with st.spinner("Preparing receptor..."):
                try:
                    rec_result = prepare_receptor_vina_batch(
                        receptor_pdb_path, work,
                        auto_cx, auto_cy, auto_cz,
                        box_sx, box_sy, box_sz,
                        keep_chain=keep_chain,
                        keep_cofactors=keep_cofactor_keys
                    )
                    st.session_state.receptor_result = rec_result
                    st.success("✅ Receptor preparation completed!")
                    
                    with st.expander("📋 Preparation Log", expanded=True):
                        st.text("\n".join(rec_result["log"]))
                except Exception as e:
                    st.error(f"❌ Receptor preparation failed: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.active_step == 4:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">Step 6 · Ligands</div>
        <div class="step-heading">Source and prepare ligands for docking</div>
    """, unsafe_allow_html=True)

    rec_result = st.session_state.get("receptor_result", None)

    if not rec_result:
        st.warning("⚠️ No receptor prepared yet. Go to the **Receptor Preparation** tab first.")

    lig_source = st.radio("",
                          ["From IMPPAT (CSV - ALL compounds)", "Manual SMILES", "Upload SDF/MOL2"],
                          horizontal=True, key="lig_source")

    selected_smiles_list = []

    if lig_source == "From IMPPAT (CSV - ALL compounds)":
        if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
            df_lig = pd.read_csv(st.session_state.csv_path)
            df_with_smi = df_lig[df_lig["SMILES_IMPPAT"].notna() & (df_lig["SMILES_IMPPAT"]!="")]
            st.info(f"📊 {len(df_with_smi)} compounds with SMILES available from IMPPAT.")

            filter_ro5 = st.checkbox("Filter by Lipinski's Rule of 5", value=False)
            if filter_ro5:
                df_with_smi = df_with_smi[df_with_smi["LIPINSKI"]=="Passed"]
                st.caption(f"{len(df_with_smi)} compounds after Lipinski filter")

            df_to_dock = df_with_smi

            if len(df_to_dock) > 0:
                st.dataframe(df_to_dock[["IMPHY_ID","CID","Phytochemical_Name","SMILES_IMPPAT","LIPINSKI"]], 
                           use_container_width=True, height=300)
                selected_smiles_list = list(zip(df_to_dock["Phytochemical_Name"], df_to_dock["SMILES_IMPPAT"]))
            else:
                st.warning("No compounds available.")
        else:
            st.warning("No IMPPAT CSV found. Run Phytochemical Extraction first (Tab 1).")

    elif lig_source == "Manual SMILES":
        manual_input = st.text_area(
            "SMILES input (one per line, optionally with name)",
            placeholder="CC(=O)Oc1ccccc1C(=O)O  Aspirin\nC1=CC(=CC=C1C2=CC(=O)C3=C(O2)C=C(C=C3O)O)O  Apigenin",
            height=150
        )
        if manual_input.strip():
            for line in manual_input.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    smi = parts[0]
                    if len(parts) > 1:
                        name = " ".join(parts[1:])
                    else:
                        name = f"lig_{len(selected_smiles_list)+1}"
                    selected_smiles_list.append((name, smi))
            st.success(f"✅ Parsed {len(selected_smiles_list)} SMILES entries.")

    else:
        uploaded_lig = st.file_uploader("Upload ligand file", type=["sdf","mol2","smi"])
        if uploaded_lig:
            tmp = unique_filepath(os.path.join(OUTPUT_DIR, f"uploaded_lig{Path(uploaded_lig.name).suffix}"))
            with open(tmp, "wb") as f: 
                f.write(uploaded_lig.getvalue())
            ext = Path(uploaded_lig.name).suffix.lower()
            
            with st.spinner(f"Processing {uploaded_lig.name}..."):
                try:
                    if ext == ".smi":
                        content = open(tmp).read()
                        for line in content.strip().splitlines():
                            if line.strip():
                                parts = line.strip().split()
                                smi = parts[0]
                                name = " ".join(parts[1:]) if len(parts)>1 else f"mol_{len(selected_smiles_list)+1}"
                                selected_smiles_list.append((name, smi))
                    else:
                        if ext == ".sdf":
                            supp = Chem.SDMolSupplier(tmp, sanitize=False, removeHs=False)
                        elif ext == ".mol2":
                            supp = Chem.MolFromMol2File(tmp, sanitize=False, removeHs=False)
                            supp = [supp] if supp else []
                        else:
                            st.error(f"Unsupported file format: {ext}")
                            supp = []
                        
                        for i, mol in enumerate(supp):
                            if mol is None:
                                continue
                            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                            smi = Chem.MolToSmiles(mol, canonical=True)
                            if smi:
                                selected_smiles_list.append((name.strip() or f"mol_{i+1}", smi))
                    st.success(f"✅ Loaded {len(selected_smiles_list)} molecules")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    if selected_smiles_list:
        st.markdown("---")
        col_prep1, col_prep2 = st.columns([3, 1])
        with col_prep1:
            st.markdown(
                f"<div style='padding:10px 0;font-size:0.93rem;color:#2e7d32;font-weight:600;'>"
                f"🧪 {len(selected_smiles_list)} ligand(s) ready for preparation</div>",
                unsafe_allow_html=True
            )
        with col_prep2:
            prep_btn = st.button("Prepare Ligands", type="primary", use_container_width=True, key="prep_ligands_btn")

        if prep_btn:
            ph_val_prep = st.session_state.get("ph_val", 7.4)
            work_prep = unique_workdir(Path(OUTPUT_DIR) / "ligprep")
            work_prep.mkdir(parents=True, exist_ok=True)
            prep_results = []
            prog_p = st.progress(0)
            stat_p = st.empty()
            for i, (lig_name, smiles) in enumerate(selected_smiles_list):
                stat_p.text(f"Preparing {i+1}/{len(selected_smiles_list)}: {lig_name}")
                prog_p.progress((i + 1) / len(selected_smiles_list))
                lig_dir = work_prep / f"lig_{safe_name(lig_name)}"
                lig_dir.mkdir(exist_ok=True)
                try:
                    lr = prepare_ligand_vina_batch(smiles, lig_name, ph_val_prep, lig_dir)
                    prep_results.append(lr)
                except Exception as e:
                    prep_results.append({"success": False, "name": lig_name, "error": str(e)})
            prog_p.empty()
            stat_p.empty()
            ok = [r for r in prep_results if r.get("success")]
            st.success(f"✅ Prepared {len(ok)}/{len(prep_results)} ligands successfully.")
            if len(ok) < len(prep_results):
                failed_names = [r["name"] for r in prep_results if not r.get("success")]
                st.warning(f"⚠️ Failed: {', '.join(failed_names)}")
            st.session_state["ligand_prep_results"] = prep_results

    st.session_state.selected_smiles_list = selected_smiles_list

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.active_step == 5:
    render_prereq_banner(2, "No receptor prepared yet — head to the **Receptor Preparation** tab first.")

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Step 7 · Redocking</div>
        <div class="step-heading">Validate the docking protocol against the co-crystal pose</div>
    """, unsafe_allow_html=True)

    rec_result = st.session_state.get("receptor_result", None)

    if not rec_result:
        st.warning("⚠️ No receptor prepared yet. Go to the **Receptor Preparation** tab first.")
    elif not st.session_state.dock_bin_path:
        st.warning("⚠️ No docking binary configured. Go to the **Docking Engine** tab first.")
    elif not st.session_state.get("selected_smiles_list"):
        st.info("ℹ️ Tip: prepare your ligands in the **Ligand Preparation** tab first — redocking validates the exact same protocol you'll use for batch docking below.")

    if rec_result and st.session_state.receptor_pdb_path and st.session_state.dock_bin_path:
            
            cocrystal_info = st.session_state.get('cocrystal_info')
            
            if cocrystal_info and cocrystal_info.get('found'):
                detected_resname = cocrystal_info['resname']
                detected_chain = cocrystal_info.get('chain') or '-'
                detected_resid = cocrystal_info.get('resid') or ''
                center = cocrystal_info.get('center') or (0.0, 0.0, 0.0)
                
                st.markdown(
                    f"""
                    <div class="ligand-detect-card">
                        <div class="ligand-detect-item">
                            <div class="ligand-detect-label">Co-crystal Ligand (from Receptor Prep)</div>
                            <div class="ligand-detect-value">✅ {detected_resname} (Chain {detected_chain}, Resid {detected_resid})</div>
                        </div>
                        <div class="ligand-detect-item">
                            <div class="ligand-detect-label">Center Coordinates</div>
                            <div class="ligand-detect-value">📍 ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                cocrystal_smiles = st.text_input(
                    "🔬 SMILES of Co-crystallized Ligand",
                    placeholder="e.g., CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
                    help="Enter the SMILES of the ligand that is co-crystallized with your receptor for RMSD validation.",
                    key="cocrystal_smiles_input"
                )
                
                if st.session_state.current_engine in ("VINA", "VINAXB"):
                    protocol_summary = (
                        f"{st.session_state.current_engine}: exhaustiveness {st.session_state.dock_params.get('exhaustiveness')}, "
                        f"modes {st.session_state.dock_params.get('num_modes')}, energy range {st.session_state.dock_params.get('energy_range')} kcal/mol"
                    )
                elif st.session_state.current_engine == "GNINA":
                    protocol_summary = (
                        f"GNINA: exhaustiveness {st.session_state.dock_params.get('gnina_exhaustiveness')}, "
                        f"modes {st.session_state.dock_params.get('gnina_num_modes')}, CNN scoring {st.session_state.dock_params.get('gnina_cnn_scoring')}"
                    )
                else:
                    protocol_summary = (
                        f"Uni-Dock: search mode {st.session_state.dock_params.get('unidock_search_mode')}, "
                        f"modes {st.session_state.dock_params.get('unidock_num_modes')}, scoring {st.session_state.dock_params.get('unidock_scoring')}"
                    )
                st.info(f"🔁 Redocking uses the same protocol as batch docking: {protocol_summary}")

                with st.expander("⚙️ Redocking accuracy options", expanded=True):
                    use_crystal_box = st.checkbox(
                        "Use compact box around the crystal ligand for validation",
                        value=True,
                        help="Keeps redocking focused on the known binding site.",
                        key="use_crystal_box"
                    )
                    rb1, rb2, rb3 = st.columns(3)
                    with rb1:
                        redock_padding = st.slider("Box padding (Å)", 4.0, 12.0, 8.0, 0.5, key="redock_padding")
                    with rb2:
                        redock_min_box = st.slider("Minimum box side (Å)", 8.0, 18.0, 12.0, 0.5, key="redock_min_box")
                    with rb3:
                        redock_max_box = st.slider("Maximum box side (Å)", 16.0, 32.0, 24.0, 0.5, key="redock_max_box")
                
                if st.button("Run Redocking", type="secondary", use_container_width=True, key="redock_button"):
                    if not cocrystal_smiles.strip():
                        st.error("❌ Please enter the SMILES string of the co-crystal ligand.")
                    elif not st.session_state.dock_bin_path:
                        st.error("❌ Please find/configure the docking binary first using the 'Find Binary' button above.")
                    else:
                        redock_params = dict(st.session_state.dock_params)
                        redock_params["log_placeholder"] = None
                        redock_params["use_crystal_box"] = use_crystal_box
                        redock_params["redock_box_padding"] = redock_padding
                        redock_params["redock_min_box_size"] = redock_min_box
                        redock_params["redock_max_box_size"] = redock_max_box
                        work = unique_workdir(Path(OUTPUT_DIR) / "redock_validation")
                        work.mkdir(parents=True, exist_ok=True)
                        
                        with st.spinner(f"Running redocking validation with {st.session_state.current_engine}"):
                            redock_result = redock_cocrystal_ligand_from_smiles(
                                rec_result, 
                                cocrystal_smiles.strip(), 
                                detected_resname,
                                st.session_state.current_engine, 
                                st.session_state.dock_bin_path, 
                                redock_params, 
                                work,
                                st.session_state.receptor_pdb_path,
                                cocrystal_info.get('chain'),
                                cocrystal_info.get('resid')
                            )
                        
                        if redock_result and redock_result['success']:
                            st.session_state.redock_result = redock_result
                        else:
                            st.session_state.redock_result = None
                            st.error("Redocking validation failed. Check SMILES and residue name.")

                redock_result = st.session_state.get("redock_result")
                if redock_result and redock_result.get('success'):
                    rmsd_val = redock_result['rmsd']
                    message, status = get_rmsd_validation_message(rmsd_val)

                    if status == "excellent":
                        st.success(f"🎉 {message}")
                    elif status == "good":
                        st.success(f"✅ {message}")
                    elif status == "acceptable":
                        st.warning(f"⚠️ {message}")
                    else:
                        st.error(f"❌ {message}")

                    binding_affinity = redock_result.get("binding_affinity")
                    affinity_text = f"{binding_affinity:.3f}" if binding_affinity is not None else "N/A"

                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a: st.metric("RMSD (Pose 1)", f"{rmsd_val:.3f} Å" if rmsd_val is not None else "N/A")
                    with col_b: st.metric("Binding affinity (kcal/mol)", affinity_text)
                    with col_c: st.metric("Original Atoms", redock_result.get('n_atoms_original', '?'))
                    with col_d: st.metric("Docked Heavy Atoms", redock_result.get('n_atoms_docked', '?'))

                    rmsd_details = redock_result.get("rmsd_details") or {}
                    if rmsd_details:
                        pose_rows = rmsd_details.get("pose_rmsds") or []
                        if pose_rows:
                            with st.expander("Pose RMSD table (After Kabsch Alignment)"):
                                display_df = pd.DataFrame([
                                    {
                                        "Pose": str(r.get("pose", "")),
                                        "RMSD (Å)": f"{r['rmsd']:.3f}" if r.get("rmsd") is not None else "—",
                                        "Affinity (kcal/mol)": f"{r['affinity']:.3f}" if r.get("affinity") is not None else "—",
                                    }
                                    for r in pose_rows
                                ])
                                st.dataframe(display_df, use_container_width=True, hide_index=True)

                    st.markdown("#### Pose Overlay")
                    render_redocking_pose_viewer(
                        redock_result.get('original_ligand_pdb'),
                        redock_result.get('docked_sdf')
                    )

                    col_d, col_e = st.columns(2)
                    with col_d:
                        if redock_result.get('docked_sdf') and os.path.exists(redock_result['docked_sdf']):
                            with open(redock_result['docked_sdf'], "rb") as f:
                                st.download_button("Redocked SDF", data=f,
                                                   file_name=os.path.basename(redock_result['docked_sdf']),
                                                   mime="chemical/x-mdl-sdfile",
                                                   key="redock_download_sdf")
                    with col_e:
                        if redock_result.get('docked_pdbqt') and os.path.exists(redock_result['docked_pdbqt']):
                            with open(redock_result['docked_pdbqt'], "rb") as f:
                                st.download_button("Redocked PDBQT", data=f,
                                                   file_name=os.path.basename(redock_result['docked_pdbqt']),
                                                   mime="chemical/x-pdbqt",
                                                   key="redock_download_pdbqt")

                    if rmsd_val is None:
                        st.warning("💡 **Recommendation:** RMSD could not be calculated. Check that the SMILES matches the co-crystal ligand residue exactly.")
                    elif rmsd_val < 2.0:
                        st.info("💡 **Recommendation:** Your docking protocol is validated! Proceed with batch docking.")
                    elif rmsd_val < 3.0:
                        st.info("💡 **Recommendation:** Consider increasing exhaustiveness or adjusting box size.")
                    else:
                        st.warning("💡 **Recommendation:** Optimize your docking protocol before batch docking.")
            else:
                st.warning("⚠️ No co-crystal ligand detected in the Receptor Preparation step. Please prepare a receptor with a detected ligand first.")

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.active_step == 6:
    rec_result = st.session_state.receptor_result
    dock_bin_path = st.session_state.dock_bin_path
    dock_params = st.session_state.dock_params
    ph_val = st.session_state.get("ph_val", 7.4)
    selected_smiles_list = st.session_state.get("selected_smiles_list", [])
    current_engine = st.session_state.current_engine

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Step 8 · Batch Docking</div>
    """, unsafe_allow_html=True)

    _status_parts = []
    if not rec_result:
        st.warning("⚠️ No receptor prepared yet. Go to the **Receptor Preparation** tab first.")
    if not dock_bin_path:
        st.warning("⚠️ No docking binary configured. Go to the **Docking Engine** tab first.")
    if not selected_smiles_list:
        st.info("ℹ️ No ligands selected. Go to the **Ligand Preparation** tab to add ligands.")

    current_receptor_for_session = st.session_state.get("receptor_pdb_path")
    if (
        "session_name" not in st.session_state
        or st.session_state.get("session_receptor_path") != current_receptor_for_session
    ):
        st.session_state["session_name"] = default_docking_session_name(current_receptor_for_session)
        st.session_state["session_receptor_path"] = current_receptor_for_session
    session_name = st.session_state["session_name"]

    run_btn = st.button("Run Batch Docking", type="primary", use_container_width=True,
                        disabled=(len(selected_smiles_list)==0 or rec_result is None or dock_bin_path is None))

    if run_btn:
        if not rec_result:
            st.error("❌ No receptor prepared.")
        elif not selected_smiles_list:
            st.error("❌ No ligands selected.")
        elif not dock_bin_path:
            st.error("❌ No docking binary found. Please configure the engine and find binary first.")
        else:
            work = unique_workdir(Path(OUTPUT_DIR) / safe_name(session_name))
            work.mkdir(parents=True, exist_ok=True)

            ligand_results = []
            dock_records = []

            with st.spinner("Checking everything…"):
                for i, (lig_name, smiles) in enumerate(selected_smiles_list):
                    lig_dir = work / f"lig_{safe_name(lig_name)}"
                    lig_dir.mkdir(exist_ok=True)
                    try:
                        lr = prepare_ligand_vina_batch(smiles, lig_name, ph_val, lig_dir)
                        ligand_results.append(lr)
                    except Exception as e:
                        st.warning(f"⚠️ Failed for {lig_name}: {e}")

            if ligand_results:
                pass  
            else:
                st.error("❌ No ligands prepared.")
                st.stop()

            st.subheader("Docking Calculations")
            prog_dock = st.progress(0)
            dock_status = st.empty()

            for i, lr in enumerate(ligand_results):
                dock_status.text(f"Docking {i+1}/{len(ligand_results)}: {lr['name']}")
                prog_dock.progress((i+1)/len(ligand_results))

                lig_input = lr["sdf"] if current_engine == "UNIDOCK" else lr["pdbqt"]
                out_prefix = str(work / f"dock_{safe_name(lr['name'])}")
                run_params = dict(dock_params)
                run_params["log_placeholder"] = None

                try:
                    docked_pdbqt, docked_sdf, dock_log = dock_one_ligand(
                        current_engine, dock_bin_path,
                        rec_result["receptor_pdbqt"],
                        lig_input, rec_result["config_file"],
                        out_prefix, run_params
                    )
                    try:
                        score_info = extract_top_score(docked_pdbqt, docked_sdf, current_engine)
                        top_score = score_info["score"]
                        dock_records.append({
                            "name": lr["name"],
                            "docked_pdbqt": docked_pdbqt,
                            "docked_sdf": docked_sdf,
                            "top_score": top_score,
                            "log": dock_log,
                        })
                    except Exception as e:
                        dock_records.append({
                            "name": lr["name"],
                            "docked_pdbqt": docked_pdbqt,
                            "docked_sdf": docked_sdf,
                            "top_score": None,
                            "log": dock_log,
                        })
                except Exception as e:
                    dock_records.append({
                        "name": lr["name"],
                        "docked_pdbqt": None,
                        "docked_sdf": None,
                        "top_score": None,
                        "log": str(e),
                    })

            prog_dock.empty()
            dock_status.empty()

            st.session_state.ligand_results = ligand_results
            st.session_state.dock_records = dock_records

            with st.spinner("📦 Packaging results…"):
                session_files = create_docking_session_files(
                    work, session_name, rec_result, ligand_results, dock_records, current_engine,
                    st.session_state.get('redock_result')
                )
            st.session_state.session_files = session_files
            st.success("🎉 Docking complete! Results are shown below.")

    dock_records = st.session_state.dock_records
    session_files = st.session_state.session_files
    redock_result = st.session_state.redock_result
    current_engine = st.session_state.current_engine
    ligand_results = st.session_state.get("ligand_results", [])

    if dock_records:
        successful = [d for d in dock_records if d["top_score"] is not None]
        if successful:
            best = min(successful, key=lambda x: x["top_score"])
            scores = [d["top_score"] for d in successful]
            avg_score = sum(scores) / len(scores) if scores else 0

            st.markdown("### Docking Summary")
            
            if redock_result and redock_result.get('success'):
                rmsd_val = redock_result['rmsd']
                if rmsd_val is None:
                    rmsd_class = "rmsd-poor"
                    rmsd_text = "RMSD could not be calculated"
                elif rmsd_val < 1.5:
                    rmsd_class = "rmsd-excellent"
                    rmsd_text = f"RMSD = {rmsd_val:.3f} Å"
                elif rmsd_val < 2.0:
                    rmsd_class = "rmsd-good"
                    rmsd_text = f"RMSD = {rmsd_val:.3f} Å"
                else:
                    rmsd_class = "rmsd-poor"
                    rmsd_text = f"RMSD = {rmsd_val:.3f} Å"
                
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                    <strong>🔄 Redocking Validation</strong><br>
                    <span class="{rmsd_class}">{rmsd_text}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if redock_result.get('original_ligand_pdb') and redock_result.get('docked_sdf'):
                    st.markdown("#### Redocking Pose Overlay")
                    render_redocking_pose_viewer(
                        redock_result.get('original_ligand_pdb'),
                        redock_result.get('docked_sdf'),
                        height=460
                    )
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Successful", f"{len(successful)}/{len(dock_records)}")
            with col2: st.metric("Best Score", f"{best['top_score']:.2f} kcal/mol")
            with col3: st.metric("Best Compound", best["name"])

            st.markdown("---")
            st.markdown("### Score Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_score_chart(dock_records, engine=current_engine or "VINA")
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                if len(successful) > 1:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.hist(scores, bins=min(10, len(scores)), color="steelblue", alpha=0.7, edgecolor="black")
                    ax2.axvline(avg_score, color="red", linestyle="--", label=f"Average: {avg_score:.2f}")
                    ax2.axvline(best['top_score'], color="green", linestyle="--", label=f"Best: {best['top_score']:.2f}")
                    ax2.set_xlabel("Binding Energy (kcal/mol)")
                    ax2.set_ylabel("Frequency")
                    ax2.set_title("Score Distribution")
                    ax2.legend()
                    st.pyplot(fig2)
                    plt.close(fig2)

            st.markdown("### Detailed Results")
            score_rows = []
            for d in dock_records:
                row = {
                    "Compound": d["name"],
                    "Score (kcal/mol)": f"{d['top_score']:.2f}" if d["top_score"] is not None else "—",
                    "Status": "✅" if d["top_score"] is not None else "❌",
                }
                score_rows.append(row)
            score_df = pd.DataFrame(score_rows)
            st.dataframe(score_df.sort_values("Score (kcal/mol)", na_position="last"), 
                        use_container_width=True, hide_index=True)

            st.markdown("### Per-Ligand Results")

            ENGINE_FULL_NAMES = {
                "VINA": "AutoDock Vina",
                "GNINA": "GNINA",
                "UNIDOCK": "Uni-Dock",
            }

            def _build_full_docking_report(records, engine_name):
                engine_key = (engine_name or "VINA").upper()
                engine_full = ENGINE_FULL_NAMES.get(engine_key, engine_name or "VINA")
                total = len(records)
                succ = [d for d in records if d["top_score"] is not None]
                failed = total - len(succ)
                ranked = sorted(
                    records,
                    key=lambda x: (x["top_score"] is None, x["top_score"] if x["top_score"] is not None else 0),
                )

                bar = "=" * 79
                thin = "-" * 79

                lines = []
                lines.append(bar)
                lines.append("IMPPATez Docking Report".center(79))
                lines.append(bar)
                lines.append("")
                lines.append(f"Engine          : {engine_full}")
                lines.append(f"Generated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append(f"Total Ligands   : {total}")
                lines.append(f"Successful      : {len(succ)}")
                lines.append(f"Failed          : {failed}")
                lines.append("")
                lines.append(bar)
                lines.append("Binding Affinity Summary")
                lines.append(bar)
                lines.append("")
                lines.append(f" {'Rank':<6}{'Ligand':<32}{'Best Score (kcal/mol)':<24}{'Status':<10}")
                lines.append(" " + "-" * 76)
                for i, d in enumerate(ranked, 1):
                    score_str = f"{d['top_score']:.3f}" if d["top_score"] is not None else "N/A"
                    status = "SUCCESS" if d["top_score"] is not None else "FAILED"
                    lines.append(f" {str(i):<6}{d['name']:<32}{score_str:<24}{status:<10}")
                lines.append("")
                lines.append(bar)
                lines.append("Detailed Docking Results")
                lines.append(bar)
                lines.append("")

                for i, d in enumerate(ranked, 1):
                    lines.append(thin)
                    lines.append(f"[{i:02d}] {d['name']}")
                    lines.append(thin)
                    lines.append("")

                    status = "SUCCESS" if d["top_score"] is not None else "FAILED"
                    lines.append(f"Status        : {status}")
                    lines.append(f"Best Score    : {d['top_score']:.3f} kcal/mol" if d["top_score"] is not None else "Best Score    : N/A")

                    poses = []
                    if d["top_score"] is not None:
                        try:
                            poses = parse_all_poses(d.get("docked_pdbqt"), d.get("docked_sdf"), engine=engine_key)
                        except Exception:
                            poses = []
                    lines.append(f"Total Poses   : {len(poses)}")
                    lines.append("")

                    if poses:
                        cols = list(poses[0].keys())
                        header_map = {"pose": "Pose", "affinity": "Affinity (kcal/mol)", "rmsd_lb": "RMSD L.B. (Å)",
                                      "rmsd_ub": "RMSD U.B. (Å)", "CNNscore": "CNN Score", "CNNaffinity": "CNN Affinity"}
                        col_w = 22
                        lines.append(" " + "".join(header_map.get(c, c).ljust(col_w) for c in cols))
                        lines.append(" " + "-" * (col_w * len(cols)))
                        for p in poses:
                            row_bits = []
                            for c in cols:
                                v = p.get(c)
                                if v is None:
                                    row_bits.append("—")
                                elif isinstance(v, float):
                                    row_bits.append(f"{v:.3f}")
                                else:
                                    row_bits.append(str(v))
                            lines.append(" " + "".join(x.ljust(col_w) for x in row_bits))
                    elif d["top_score"] is None and d.get("log"):
                        lines.append("Error / Log:")
                        lines.append(str(d["log"])[:500])
                    lines.append("")

                return "\n".join(lines)

            full_report = _build_full_docking_report(dock_records, current_engine)
            st.download_button(
                "⬇️ Download All Per-Ligand Results (.txt)",
                data=full_report,
                file_name="per_ligand_results.txt",
                mime="text/plain",
                use_container_width=True,
            )

            ranked_for_display = sorted(
                dock_records,
                key=lambda x: (x["top_score"] is None, x["top_score"] if x["top_score"] is not None else 0),
            )
            for i, d in enumerate(ranked_for_display, 1):
                status_icon = "✅" if d["top_score"] is not None else "❌"
                score_label = f"{d['top_score']:.2f} kcal/mol" if d["top_score"] is not None else "failed"
                st.markdown(f"**{status_icon} {i}. {d['name']}** — {score_label}")

            st.markdown("---")
            st.markdown("### 🧪 ADMET Prediction — Top 10 Compounds")
            st.caption(
                "Absorption, Distribution, Metabolism, Excretion & Toxicity predictions for the "
                "10 best-scoring compounds by binding energy. Calculated locally via **RDKit** "
                "(always available) + **ADMET-AI ML** (if installed: `pip install admet-ai`)."
            )

            top10 = sorted(successful, key=lambda x: x["top_score"])[:10]
            smiles_lookup = {lr["name"]: lr.get("ph_smiles") for lr in ligand_results}

            _btn_col, _retry_col = st.columns([3, 2])
            with _btn_col:
                run_clicked = st.button("🧪 Run ADMET Prediction (Preferred Compounds)", key="btn_admet_top10", type="primary")
            with _retry_col:
                if st.button("🔄 Re-check ADMET-AI install", key="btn_admet_recheck"):
                    _load_admet_model_cached.clear()
                    st.session_state.pop("admet_top10_results", None)
                    st.rerun()
            st.caption(
                "If you just ran `pip install admet-ai`, the app process needs to reload the model — "
                "use **Re-check ADMET-AI install** above, or fully restart the Streamlit app "
                "(a browser refresh alone won't reload it)."
            )

            if run_clicked:
                admet_cache = {}
                with st.spinner(f"Calculating ADMET properties for {len(top10)} compound(s)…"):
                    for d in top10:
                        smi = smiles_lookup.get(d["name"])
                        admet_cache[d["name"]] = _calc_admet_properties(smi) if smi else {"error": "SMILES not found"}
                st.session_state.admet_top10_results = admet_cache

            admet_results = st.session_state.get("admet_top10_results")
            if admet_results:
                summary_rows = []
                for rank, d in enumerate(top10, 1):
                    p = admet_results.get(d["name"], {})
                    if "error" in p:
                        summary_rows.append({
                            "Rank": rank, "Compound": d["name"],
                            "Score (kcal/mol)": f"{d['top_score']:.2f}",
                            "MW": "—", "LogP": "—", "TPSA": "—", "HBD": "—", "HBA": "—",
                            "Lipinski": "—", "Bioavail. %": "—", "GI": "—", "BBB": "—",
                            "PAINS": "—", "BRENK": "—",
                        })
                        continue
                    summary_rows.append({
                        "Rank": rank, "Compound": d["name"],
                        "Score (kcal/mol)": f"{d['top_score']:.2f}",
                        "MW": p["mw"], "LogP": p["logp"], "TPSA": p["tpsa"],
                        "HBD": p["hbd"], "HBA": p["hba"],
                        "Lipinski": "Pass" if p["lipinski_pass"] else "Fail",
                        "Bioavail. %": int(p["bioavailability_score"] * 100),
                        "GI": p["gi_absorption"], "BBB": p["bbb"],
                        "PAINS": len(p["pains_alerts"]), "BRENK": len(p["brenk_alerts"]),
                    })
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇️ Download ADMET Summary (.csv)",
                    data=summary_df.to_csv(index=False),
                    file_name="admet_top10_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.markdown("#### Per-Compound Details")
                for rank, d in enumerate(top10, 1):
                    p = admet_results.get(d["name"], {})
                    with st.expander(f"{rank}. {d['name']} — {d['top_score']:.2f} kcal/mol"):
                        _render_admet_compound(p, d["name"], d["top_score"])
            else:
                st.info("Click **Run ADMET Prediction (Preferred Compounds)** to compute ADMET properties for the most preferable / best-scoring compounds.")

        st.markdown("---")
        st.markdown("### Download Results")

        if session_files and os.path.exists(session_files["zip"]):
            zip_size = os.path.getsize(session_files["zip"])
            with open(session_files["zip"], "rb") as f:
                st.download_button(
                    f"Download ZIP ({zip_size/1024/1024:.1f} MB)",
                    data=f,
                    file_name=os.path.basename(session_files["zip"]),
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                )
            st.caption("ZIP contains: receptor PDBQT · ligand SDFs & PDBQTs · docked poses · summary CSV · session report")

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.markdown("""
<div class="footer-bar">
    <div class="footer-brand">🌿 IMPPAT<span>ez</span></div>
    <div class="footer-meta">
        Developed by
        <a href="https://www.linkedin.com/in/mdnabiulhoque/"
           target="_blank"
           style="font-weight:700; color:inherit; text-decoration:none;">
            Nabiul Orko
        </a><br>
        <span style="font-size:0.72rem; color:#6a8a9e;">
            RDKit · Vina · GNINA · Uni-Dock · OpenBabel · ProDy · Meeko
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
