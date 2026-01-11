import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from astroquery.vizier import Vizier
except Exception:
    Vizier = None

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except Exception:
    SkyCoord = None
    u = None

try:
    from astroquery.sdss import SDSS
except Exception:
    SDSS = None

# Add project root to path for logger
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from scripts.utils.logger import print_status
except Exception:

    def print_status(msg: str, level: str = "INFO"):
        print(f"[{level}] {msg}")


VIZIER_PGC_CATALOG = "VII/237/pgc"
VIZIER_HI_CATALOG = "VII/238/hidat"
VIZIER_HO2009_CATALOG = "J/ApJS/183/1/table1"
VIZIER_HIPASS_CATALOG = "VIII/73/hicat"
VIZIER_6DFGS_FP_CATALOG = "J/MNRAS/443/1231/table2"
VIZIER_HO2007_CATALOG = "J/ApJ/668/94/sample"
VIZIER_BASSDR2_CATALOG = "J/ApJS/261/6/best"
VIZIER_APJ929_84_CATALOG = "J/ApJ/929/84/sample"
VIZIER_MNRAS482_1427_CATALOG = "J/MNRAS/482/1427/table1"

DEFAULT_LIT_OVERRIDES_CSV = str(
    Path(__file__).resolve().parents[2] / "data" / "raw" / "external" / "sigma_literature_overrides.csv"
)

LEDA_URL = "http://atlas.obs-hp.fr/hyperleda/ledacat.cgi"


@dataclass
class SigmaCandidate:
    value_kms: Optional[float]
    error_kms: Optional[float]
    source: str
    method: str
    details: Dict[str, Any]


def _is_finite_pos(x: Optional[float]) -> bool:
    return x is not None and np.isfinite(x) and float(x) > 0


def _try_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if hasattr(x, "mask") and bool(x.mask):
            return None
        return float(x)
    except Exception:
        return None


def load_literature_overrides(csv_path: str) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    try:
        p = Path(str(csv_path)).expanduser()
        if not p.exists():
            return overrides

        df = pd.read_csv(p)
        if df is None or len(df) == 0:
            return overrides

        for _, r in df.iterrows():
            g = str(r.get("galaxy", "")).strip()
            if not g:
                continue
            sig = _try_float(r.get("sigma_kms"))
            if not _is_finite_pos(sig):
                continue
            err = _try_float(r.get("error_kms"))
            refcode = str(r.get("refcode", "")).strip()
            note = str(r.get("note", "")).strip()
            overrides[g] = {
                "sigma_kms": float(sig),
                "error_kms": float(err) if _is_finite_pos(err) else None,
                "refcode": refcode,
                "note": note,
                "csv_path": str(p),
            }
        return overrides
    except Exception:
        return overrides


def query_apj929_84_sigma(coord: Optional[SkyCoord], radius_arcsec: float = 60.0) -> Optional[SigmaCandidate]:
    if Vizier is None or coord is None or SkyCoord is None or u is None:
        return None

    try:
        v = Vizier(columns=["Name", "Sigma", "RAJ2000", "DEJ2000"])
        v.ROW_LIMIT = 20
        cats = v.query_region(coord, radius=float(radius_arcsec) * u.arcsec, catalog=VIZIER_APJ929_84_CATALOG)
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None

        t = cats[0]

        best_idx = 0
        best_sep = None
        for i in range(len(t)):
            ra = _try_float(t["RAJ2000"][i])
            dec = _try_float(t["DEJ2000"][i])
            if ra is None or dec is None:
                continue
            c2 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            sep = coord.separation(c2).arcsec
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_idx = i

        sig = _try_float(t["Sigma"][best_idx])
        if not _is_finite_pos(sig):
            return None

        name = str(t["Name"][best_idx]) if "Name" in t.colnames else ""

        return SigmaCandidate(
            value_kms=float(sig),
            error_kms=None,
            source="ApJ 929:84 (VizieR J/ApJ/929/84)",
            method="stellar absorption",
            details={
                "catalog": VIZIER_APJ929_84_CATALOG,
                "Name": name,
                "Sigma": float(sig),
                "sep_arcsec": float(best_sep) if best_sep is not None else None,
            },
        )
    except Exception:
        return None


def query_mnras482_1427_sigma(coord: Optional[SkyCoord], radius_arcsec: float = 60.0) -> Optional[SigmaCandidate]:
    if Vizier is None or coord is None or SkyCoord is None or u is None:
        return None

    try:
        v = Vizier(
            columns=[
                "Name",
                "SimbadName",
                "_RA",
                "_DE",
                "Vdisp30",
                "e_Vdisp30",
                "Vdisp30Hd",
                "e_Vdisp30Hd",
                "Vdisp5",
                "e_Vdisp5",
            ]
        )
        v.ROW_LIMIT = 20
        cats = v.query_region(coord, radius=float(radius_arcsec) * u.arcsec, catalog=VIZIER_MNRAS482_1427_CATALOG)
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None

        t = cats[0]

        best_idx = 0
        best_sep = None
        for i in range(len(t)):
            ra = _try_float(t["_RA"][i])
            dec = _try_float(t["_DE"][i])
            if ra is None or dec is None:
                continue
            c2 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            sep = coord.separation(c2).arcsec
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_idx = i

        vdisp30 = _try_float(t["Vdisp30"][best_idx])
        e_vdisp30 = _try_float(t["e_Vdisp30"][best_idx])
        vdisp30hd = _try_float(t["Vdisp30Hd"][best_idx])
        e_vdisp30hd = _try_float(t["e_Vdisp30Hd"][best_idx])

        chosen_val = vdisp30hd if _is_finite_pos(vdisp30hd) else vdisp30
        chosen_err = e_vdisp30hd if _is_finite_pos(vdisp30hd) else e_vdisp30
        chosen_key = "Vdisp30Hd" if _is_finite_pos(vdisp30hd) else "Vdisp30"

        if not _is_finite_pos(chosen_val):
            return None

        name = str(t["Name"][best_idx]) if "Name" in t.colnames else ""
        sname = str(t["SimbadName"][best_idx]) if "SimbadName" in t.colnames else ""
        vdisp5 = _try_float(t["Vdisp5"][best_idx])
        e_vdisp5 = _try_float(t["e_Vdisp5"][best_idx])

        return SigmaCandidate(
            value_kms=float(chosen_val),
            error_kms=float(chosen_err) if _is_finite_pos(chosen_err) else None,
            source="MNRAS 482:1427 (VizieR J/MNRAS/482/1427)",
            method="stellar absorption",
            details={
                "catalog": VIZIER_MNRAS482_1427_CATALOG,
                "Name": name,
                "SimbadName": sname,
                "chosen": chosen_key,
                "Vdisp30": float(vdisp30) if vdisp30 is not None and np.isfinite(vdisp30) else None,
                "e_Vdisp30": float(e_vdisp30) if e_vdisp30 is not None and np.isfinite(e_vdisp30) else None,
                "Vdisp30Hd": float(vdisp30hd) if vdisp30hd is not None and np.isfinite(vdisp30hd) else None,
                "e_Vdisp30Hd": float(e_vdisp30hd) if e_vdisp30hd is not None and np.isfinite(e_vdisp30hd) else None,
                "Vdisp5": float(vdisp5) if vdisp5 is not None and np.isfinite(vdisp5) else None,
                "e_Vdisp5": float(e_vdisp5) if e_vdisp5 is not None and np.isfinite(e_vdisp5) else None,
                "sep_arcsec": float(best_sep) if best_sep is not None else None,
            },
        )
    except Exception:
        return None


def query_ho2007_sigma(coord: Optional[SkyCoord], radius_arcsec: float = 60.0) -> Optional[SigmaCandidate]:
    if Vizier is None or coord is None or SkyCoord is None or u is None:
        return None

    try:
        v = Vizier(columns=["Name", "sig0", "_RA", "_DE", "Vm", "W50", "T", "Bar?"])
        v.ROW_LIMIT = 20
        cats = v.query_region(coord, radius=float(radius_arcsec) * u.arcsec, catalog=VIZIER_HO2007_CATALOG)
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None

        t = cats[0]

        best_idx = 0
        best_sep = None
        for i in range(len(t)):
            ra = _try_float(t["_RA"][i])
            dec = _try_float(t["_DE"][i])
            if ra is None or dec is None:
                continue
            c2 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            sep = coord.separation(c2).arcsec
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_idx = i

        sig0 = _try_float(t["sig0"][best_idx])
        if not _is_finite_pos(sig0):
            return None

        name = str(t["Name"][best_idx]) if "Name" in t.colnames else ""
        vm = _try_float(t["Vm"][best_idx]) if "Vm" in t.colnames else None
        w50 = _try_float(t["W50"][best_idx]) if "W50" in t.colnames else None
        ttype = _try_float(t["T"][best_idx]) if "T" in t.colnames else None
        bar = str(t["Bar?"][best_idx]) if "Bar?" in t.colnames else ""

        return SigmaCandidate(
            value_kms=float(sig0),
            error_kms=None,
            source="Ho 2007 (J/ApJ/668/94)",
            method="stellar absorption",
            details={
                "catalog": VIZIER_HO2007_CATALOG,
                "name": name,
                "sig0": float(sig0),
                "Vm": float(vm) if vm is not None and np.isfinite(vm) else None,
                "W50": float(w50) if w50 is not None and np.isfinite(w50) else None,
                "T": float(ttype) if ttype is not None and np.isfinite(ttype) else None,
                "bar": bar,
                "sep_arcsec": float(best_sep) if best_sep is not None else None,
            },
        )
    except Exception:
        return None


def query_bassdr2_sigma(coord: Optional[SkyCoord], radius_arcsec: float = 60.0) -> Optional[SigmaCandidate]:
    if Vizier is None or coord is None or SkyCoord is None or u is None:
        return None

    try:
        v = Vizier(columns=["sigma", "e_sigma", "sigmaS", "e_sigmaS", "sigmaCaT", "e_sigmaCaT", "SimbadName", "_RA", "_DE"])
        v.ROW_LIMIT = 20
        cats = v.query_region(coord, radius=float(radius_arcsec) * u.arcsec, catalog=VIZIER_BASSDR2_CATALOG)
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None

        t = cats[0]

        best_idx = 0
        best_sep = None
        for i in range(len(t)):
            ra = _try_float(t["_RA"][i])
            dec = _try_float(t["_DE"][i])
            if ra is None or dec is None:
                continue
            c2 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            sep = coord.separation(c2).arcsec
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_idx = i

        sig = _try_float(t["sigma"][best_idx])
        if not _is_finite_pos(sig):
            return None

        e_sig = _try_float(t["e_sigma"][best_idx])
        sigS = _try_float(t["sigmaS"][best_idx])
        e_sigS = _try_float(t["e_sigmaS"][best_idx])
        sigCaT = _try_float(t["sigmaCaT"][best_idx])
        e_sigCaT = _try_float(t["e_sigmaCaT"][best_idx])
        sname = str(t["SimbadName"][best_idx]) if "SimbadName" in t.colnames else ""

        return SigmaCandidate(
            value_kms=float(sig),
            error_kms=float(e_sig) if _is_finite_pos(e_sig) else None,
            source="BASS DR2 (Koss+2022; J/ApJS/261/6)",
            method="stellar absorption",
            details={
                "catalog": VIZIER_BASSDR2_CATALOG,
                "SimbadName": sname,
                "sigma": float(sig),
                "e_sigma": float(e_sig) if _is_finite_pos(e_sig) else None,
                "sigmaS": float(sigS) if sigS is not None and np.isfinite(sigS) else None,
                "e_sigmaS": float(e_sigS) if e_sigS is not None and np.isfinite(e_sigS) else None,
                "sigmaCaT": float(sigCaT) if sigCaT is not None and np.isfinite(sigCaT) else None,
                "e_sigmaCaT": float(e_sigCaT) if e_sigCaT is not None and np.isfinite(e_sigCaT) else None,
                "sep_arcsec": float(best_sep) if best_sep is not None else None,
            },
        )
    except Exception:
        return None


def query_6dfgs_fp_sigma(coord: Optional[SkyCoord], radius_arcsec: float = 60.0) -> Optional[SigmaCandidate]:
    if Vizier is None or coord is None or SkyCoord is None or u is None:
        return None

    try:
        v = Vizier(columns=["6dFGS", "Vd", "e_Vd", "_RA", "_DE", "z", "S/N", "R"])
        v.ROW_LIMIT = 20
        cats = v.query_region(coord, radius=float(radius_arcsec) * u.arcsec, catalog=VIZIER_6DFGS_FP_CATALOG)
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None

        t = cats[0]

        best_idx = 0
        best_sep = None
        for i in range(len(t)):
            ra = _try_float(t["_RA"][i])
            dec = _try_float(t["_DE"][i])
            if ra is None or dec is None:
                continue
            c2 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            sep = coord.separation(c2).arcsec
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_idx = i

        vd = _try_float(t["Vd"][best_idx])
        if not _is_finite_pos(vd):
            return None

        e_vd = _try_float(t["e_Vd"][best_idx])
        sixdfgs = str(t["6dFGS"][best_idx]) if "6dFGS" in t.colnames else ""
        z = _try_float(t["z"][best_idx])
        sn = _try_float(t["S/N"][best_idx]) if "S/N" in t.colnames else None
        r = _try_float(t["R"][best_idx])

        return SigmaCandidate(
            value_kms=float(vd),
            error_kms=float(e_vd) if _is_finite_pos(e_vd) else None,
            source="6dFGSv FP (Campbell+2014; J/MNRAS/443/1231)",
            method="stellar absorption",
            details={
                "catalog": VIZIER_6DFGS_FP_CATALOG,
                "6dFGS": sixdfgs,
                "Vd": float(vd),
                "e_Vd": float(e_vd) if _is_finite_pos(e_vd) else None,
                "z": float(z) if z is not None and np.isfinite(z) else None,
                "SNR": float(sn) if sn is not None and np.isfinite(sn) else None,
                "R": float(r) if r is not None and np.isfinite(r) else None,
                "sep_arcsec": float(best_sep) if best_sep is not None else None,
            },
        )
    except Exception:
        return None


def sigma_from_w50(w50: float) -> Optional[float]:
    if not _is_finite_pos(w50):
        return None
    return 0.35 * float(w50)


def _norm_name(s: str) -> str:
    return str(s).strip().upper().replace(" ", "").replace("NGC0", "NGC")


def resolve_pgc(name: str) -> Tuple[Optional[int], Optional[SkyCoord]]:
    if Vizier is None:
        return None, None

    try:
        v = Vizier(columns=["PGC", "RAJ2000", "DEJ2000"])
        v.ROW_LIMIT = 1
        cats = v.query_object(str(name).strip(), catalog=VIZIER_PGC_CATALOG)
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None, None

        t = cats[0]
        pgc = _try_float(t["PGC"][0])
        ra = t["RAJ2000"][0]
        dec = t["DEJ2000"][0]

        coord = None
        if SkyCoord is not None:
            try:
                # Catalog returns degrees for VII/237/pgc, but accept sexagesimal just in case
                coord = SkyCoord(str(ra), str(dec), unit=(u.deg, u.deg), frame="icrs")
            except Exception:
                try:
                    coord = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg), frame="icrs")
                except Exception:
                    coord = None

        return (int(pgc) if pgc is not None else None), coord
    except Exception:
        return None, None


def query_leda_single_field(obj: str, field: str, timeout: float = 10.0) -> Optional[float]:
    params = {"o": str(obj).strip(), "d": str(field).strip(), "a": "t"}

    try:
        r = requests.get(LEDA_URL, params=params, timeout=timeout)
        r.raise_for_status()

        if "Internal database error" in r.text:
            return None

        for line in r.text.split("\n"):
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("<"):
                continue
            tokens = s.split()
            if not tokens:
                continue
            raw = tokens[-1]
            val = _try_float(raw)
            if val is not None:
                return val
            if isinstance(raw, str):
                cleaned = raw.strip().strip("()")
                val = _try_float(cleaned)
                if val is not None:
                    return val
            return None
        return None
    except Exception:
        return None


def query_hi_log2vm(pgc: int) -> Optional[float]:
    if Vizier is None or pgc is None:
        return None

    try:
        v = Vizier(columns=["PGC", "log(2Vm)"])
        v.ROW_LIMIT = 1
        cats = v.query_constraints(catalog=VIZIER_HI_CATALOG, PGC=int(pgc))
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None
        return _try_float(cats[0]["log(2Vm)"][0])
    except Exception:
        return None


def sigma_from_log2vm(log2vm: float, mode: str) -> Optional[float]:
    if not _is_finite_pos(log2vm):
        return None

    # HyperLEDA HI table provides log10(2*Vmax) in km/s.
    # Thus: 2*Vmax = 10**log2vm; and W50 ~ 2*Vmax.
    w50 = 10 ** float(log2vm)
    vmax = 0.5 * w50

    if mode == "kh13_w50":
        return 0.35 * w50

    if mode == "calibrated_vmax":
        # Empirical calibration previously used in this project history
        return 0.467 * vmax + 42.931

    raise ValueError(f"Unknown HI proxy mode: {mode}")


def query_ho2009_sigma(name: str, coord: Optional[SkyCoord]) -> Optional[SigmaCandidate]:
    if Vizier is None:
        return None

    try:
        v = Vizier(columns=["Name", "sig", "e_sig", "l_sig", "_RA", "_DE", "Simbad", "NED", "LEDA"])
        v.ROW_LIMIT = 10
        cats = v.query_object(str(name).strip(), catalog=VIZIER_HO2009_CATALOG)
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None

        t = cats[0]

        best_idx = 0
        if coord is not None and SkyCoord is not None:
            best_sep = None
            for i in range(len(t)):
                ra = _try_float(t["_RA"][i])
                dec = _try_float(t["_DE"][i])
                if ra is None or dec is None:
                    continue
                c2 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                sep = coord.separation(c2).arcsec
                if best_sep is None or sep < best_sep:
                    best_sep = sep
                    best_idx = i

        sig = _try_float(t["sig"][best_idx])
        if not _is_finite_pos(sig):
            return None

        e_sig = _try_float(t["e_sig"][best_idx])

        return SigmaCandidate(
            value_kms=float(sig),
            error_kms=float(e_sig) if e_sig is not None and np.isfinite(e_sig) and e_sig > 0 else None,
            source="Ho+2009 (J/ApJS/183/1)",
            method="stellar absorption",
            details={
                "catalog": VIZIER_HO2009_CATALOG,
                "matched_row": int(best_idx),
                "table_name": str(t["Name"][best_idx]) if "Name" in t.colnames else None,
                "l_sig": str(t["l_sig"][best_idx]) if "l_sig" in t.colnames else None,
                "Simbad": str(t["Simbad"][best_idx]) if "Simbad" in t.colnames else None,
                "NED": str(t["NED"][best_idx]) if "NED" in t.colnames else None,
                "LEDA": str(t["LEDA"][best_idx]) if "LEDA" in t.colnames else None,
            },
        )
    except Exception:
        return None


def query_sdss_sigma(coord: Optional[SkyCoord]) -> Optional[SigmaCandidate]:
    if SDSS is None or coord is None or SkyCoord is None:
        return None

    try:
        # Query a small region and pick the closest spectroscopic galaxy with velDisp
        res = SDSS.query_region(
            coord,
            radius=5 * u.arcsec,
            spectro=True,
            fields=["ra", "dec", "class", "z", "velDisp", "velDispErr"],
        )
        if res is None or len(res) == 0:
            return None

        best_row = None
        best_sep = None
        for row in res:
            if str(row.get("class", "")).strip().upper() != "GALAXY":
                continue
            vdisp = _try_float(row.get("velDisp"))
            if not _is_finite_pos(vdisp):
                continue
            ra = _try_float(row.get("ra"))
            dec = _try_float(row.get("dec"))
            if ra is None or dec is None:
                continue
            c2 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            sep = coord.separation(c2).arcsec
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_row = row

        if best_row is None:
            return None

        vdisp = _try_float(best_row.get("velDisp"))
        e_vdisp = _try_float(best_row.get("velDispErr"))
        z = _try_float(best_row.get("z"))

        return SigmaCandidate(
            value_kms=float(vdisp),
            error_kms=float(e_vdisp) if e_vdisp is not None and np.isfinite(e_vdisp) and e_vdisp > 0 else None,
            source="SDSS (astroquery.sdss)",
            method="stellar absorption",
            details={
                "query_radius_arcsec": 5.0,
                "separation_arcsec": float(best_sep) if best_sep is not None else None,
                "z": float(z) if z is not None else None,
            },
        )
    except Exception:
        return None


def query_hipass_w50(coord: Optional[SkyCoord], claimed_name: str) -> Optional[SigmaCandidate]:
    if Vizier is None or coord is None or SkyCoord is None:
        return None

    try:
        v = Vizier(columns=["HIPASS", "SimbadName", "W50max", "W20max", "Vhel", "FHI", "_RAJ2000", "_DEJ2000"])
        v.ROW_LIMIT = 50
        cats = v.query_region(coord, radius=10 * u.arcmin, catalog=VIZIER_HIPASS_CATALOG)
        if not cats or len(cats) == 0 or len(cats[0]) == 0:
            return None

        t = cats[0]

        best_idx = None
        best_sep = None
        for i in range(len(t)):
            ra = _try_float(t["_RAJ2000"][i])
            dec = _try_float(t["_DEJ2000"][i])
            w50 = _try_float(t["W50max"][i])
            if ra is None or dec is None or not _is_finite_pos(w50):
                continue

            c2 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            sep = coord.separation(c2).arcsec
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_idx = i

        if best_idx is None:
            return None

        w50 = _try_float(t["W50max"][best_idx])
        if not _is_finite_pos(w50):
            return None

        sigma = 0.35 * float(w50)

        return SigmaCandidate(
            value_kms=float(sigma),
            error_kms=None,
            source="HIPASS (VIII/73/hicat)",
            method="HI linewidth proxy",
            details={
                "catalog": VIZIER_HIPASS_CATALOG,
                "closest_sep_arcsec": float(best_sep) if best_sep is not None else None,
                "hipass_id": str(t["HIPASS"][best_idx]) if "HIPASS" in t.colnames else None,
                "simbad_name": str(t["SimbadName"][best_idx]) if "SimbadName" in t.colnames else None,
                "w50max": float(w50),
                "w20max": _try_float(t["W20max"][best_idx]),
                "vhel": _try_float(t["Vhel"][best_idx]),
                "fhi": _try_float(t["FHI"][best_idx]),
                "requested_object": str(claimed_name),
            },
        )
    except Exception:
        return None


def choose_best_candidate(cands: Dict[str, SigmaCandidate]) -> Optional[SigmaCandidate]:
    # Priority order chosen for reproducibility + directness.
    priority = [
        "lit_override",
        "ho2009",
        "sdss",
        "mnras482_1427",
        "apj929_84",
        "bassdr2",
        "ho2007",
        "leda_vdis",
        "sixdfgs_fp",
        "hi_log2vm",
        "leda_vrot_proxy",
        "leda_w50_proxy",
        "hipass",
    ]

    for key in priority:
        cand = cands.get(key)
        if cand is not None and _is_finite_pos(cand.value_kms):
            return cand

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild velocity dispersion table from public catalogs with explicit provenance.")
    ap.add_argument(
        "--hosts-csv",
        default=str(Path(__file__).resolve().parents[2] / "data" / "interim" / "hosts_coords.csv"),
    )
    ap.add_argument(
        "--out-csv",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "raw"
            / "external"
            / "velocity_dispersions_literature_regenerated.csv"
        ),
    )
    ap.add_argument(
        "--report-json",
        default=str(Path(__file__).resolve().parents[2] / "results" / "outputs" / "sigma_regeneration_report.json"),
    )
    ap.add_argument(
        "--hi-proxy",
        choices=["kh13_w50", "calibrated_vmax"],
        default="calibrated_vmax",
        help="How to map HyperLEDA HI log(2Vm) to sigma.",
    )
    ap.add_argument(
        "--use-ledacat",
        action="store_true",
        help="If set, query HyperLEDA ledacat.cgi for vdis/e_vdis/vrot in addition to VizieR.",
    )
    ap.add_argument(
        "--use-6dfgs",
        action="store_true",
        help="If set, query the 6dFGSv Fundamental Plane catalog for stellar velocity dispersions.",
    )
    ap.add_argument(
        "--use-ho2007",
        action="store_true",
        help="If set, query Ho 2007 kinematics catalog for central stellar velocity dispersions.",
    )
    ap.add_argument(
        "--use-bassdr2",
        action="store_true",
        help="If set, query BASS DR2 stellar velocity dispersions.",
    )
    ap.add_argument(
        "--use-apj929",
        action="store_true",
        help="If set, query VizieR J/ApJ/929/84 for stellar velocity dispersions.",
    )
    ap.add_argument(
        "--use-mnras482",
        action="store_true",
        help="If set, query VizieR J/MNRAS/482/1427 for stellar velocity dispersions.",
    )
    ap.add_argument(
        "--use-lit-overrides",
        action="store_true",
        help="If set, apply manual literature sigma overrides from a CSV file.",
    )
    ap.add_argument(
        "--lit-overrides-csv",
        default=DEFAULT_LIT_OVERRIDES_CSV,
    )
    ap.add_argument("--sleep", type=float, default=0.4, help="Delay between external queries (seconds).")
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print per-galaxy progress and which catalog/method was selected.",
    )
    ap.add_argument(
        "--verbose-every",
        type=int,
        default=1,
        help="Only print verbose progress every N galaxies (default: 1 = every galaxy).",
    )

    args = ap.parse_args()

    hosts_path = Path(args.hosts_csv)
    out_csv = Path(args.out_csv)
    report_json = Path(args.report_json)

    if not hosts_path.exists():
        print_status(f"Missing hosts file: {hosts_path}", "ERROR")
        return 2

    if Vizier is None:
        print_status("astroquery.vizier not available; cannot rebuild catalog.", "ERROR")
        return 2

    if SkyCoord is None:
        print_status("astropy not available; cannot do coordinate matching.", "ERROR")
        return 2

    hosts = pd.read_csv(hosts_path)
    n_hosts_raw = int(len(hosts))
    for col in ["source_id", "normalized_name", "ra", "dec"]:
        if col in hosts.columns:
            hosts[col] = hosts[col].astype(str).str.strip()

    if "normalized_name" in hosts.columns:
        hosts = hosts.drop_duplicates(subset=["normalized_name"], keep="first").copy()
    n_hosts_deduped = int(len(hosts))
    if n_hosts_deduped != n_hosts_raw:
        print_status(
            f"Deduplicated hosts by normalized_name: {n_hosts_raw} -> {n_hosts_deduped} (e.g., multiple anchor entries)",
            "WARNING",
        )

    print_status(f"Loaded {len(hosts)} hosts from {hosts_path}", "INFO")

    lit_overrides = load_literature_overrides(args.lit_overrides_csv) if args.use_lit_overrides else {}

    rows = []
    diagnostics = []

    n_hosts = int(len(hosts))
    for idx, h in hosts.iterrows():
        galaxy = str(h["normalized_name"]).strip()
        ra = float(h["ra"])
        dec = float(h["dec"])
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        t0 = time.time()

        pgc, _ = resolve_pgc(galaxy)
        leda_obj = f"PGC {pgc}" if pgc is not None else galaxy

        cands: Dict[str, SigmaCandidate] = {}

        if galaxy in lit_overrides:
            o = lit_overrides[galaxy]
            cands["lit_override"] = SigmaCandidate(
                value_kms=float(o["sigma_kms"]),
                error_kms=o.get("error_kms"),
                source="Literature override (manual)",
                method="stellar absorption",
                details={
                    "refcode": o.get("refcode", ""),
                    "note": o.get("note", ""),
                    "csv_path": o.get("csv_path", ""),
                },
            )

        # Ho+2009 direct sigma
        ho = query_ho2009_sigma(galaxy, coord)
        if ho is not None:
            cands["ho2009"] = ho

        # SDSS direct sigma (if available)
        sdss = query_sdss_sigma(coord)
        if sdss is not None:
            cands["sdss"] = sdss

        if args.use_6dfgs:
            sixd = query_6dfgs_fp_sigma(coord)
            if sixd is not None:
                cands["sixdfgs_fp"] = sixd

        if args.use_ho2007:
            ho07 = query_ho2007_sigma(coord)
            if ho07 is not None:
                cands["ho2007"] = ho07

        if args.use_bassdr2:
            bass = query_bassdr2_sigma(coord)
            if bass is not None:
                cands["bassdr2"] = bass

        if args.use_apj929:
            apj = query_apj929_84_sigma(coord)
            if apj is not None:
                cands["apj929_84"] = apj

        if args.use_mnras482:
            mn = query_mnras482_1427_sigma(coord)
            if mn is not None:
                cands["mnras482_1427"] = mn

        if args.use_ledacat:
            def _leda_query(field: str) -> Optional[float]:
                for o in (leda_obj, galaxy):
                    v = query_leda_single_field(o, field)
                    if v is not None and np.isfinite(v):
                        return float(v)
                return None

            leda_vdis = _leda_query("vdis")
            leda_e = _leda_query("e_vdis")
            if _is_finite_pos(leda_vdis):
                cands["leda_vdis"] = SigmaCandidate(
                    value_kms=float(leda_vdis),
                    error_kms=float(leda_e) if _is_finite_pos(leda_e) else None,
                    source="HyperLEDA (ledacat.cgi)",
                    method="stellar absorption",
                    details={"leda_obj": leda_obj, "field": "vdis", "e_field": "e_vdis"},
                )

        # HyperLEDA HI log(2Vm) proxy (VizieR)
        if pgc is not None:
            log2vm = query_hi_log2vm(pgc)
            sig_hi = sigma_from_log2vm(log2vm, mode=args.hi_proxy) if log2vm is not None else None
            if _is_finite_pos(sig_hi):
                cands["hi_log2vm"] = SigmaCandidate(
                    value_kms=float(sig_hi),
                    error_kms=None,
                    source="HyperLEDA HI (VII/238/hidat)",
                    method=f"HI proxy ({args.hi_proxy})",
                    details={"pgc": int(pgc), "log2vm": float(log2vm), "mode": args.hi_proxy},
                )
        else:
            log2vm = None

        # HIPASS W50 proxy
        hipass = query_hipass_w50(coord, galaxy)
        if hipass is not None:
            cands["hipass"] = hipass

        if args.use_ledacat and "leda_vdis" not in cands:
            leda_vrot = _leda_query("vrot")
            if _is_finite_pos(leda_vrot):
                cands["leda_vrot_proxy"] = SigmaCandidate(
                    value_kms=float(leda_vrot) / 1.7,
                    error_kms=None,
                    source="HyperLEDA (ledacat.cgi)",
                    method="vrot/1.7 proxy",
                    details={"leda_obj": leda_obj, "field": "vrot", "vrot": float(leda_vrot)},
                )

            leda_w50 = _leda_query("w50")
            sig_w50 = sigma_from_w50(leda_w50) if leda_w50 is not None else None
            if _is_finite_pos(sig_w50):
                cands["leda_w50_proxy"] = SigmaCandidate(
                    value_kms=float(sig_w50),
                    error_kms=None,
                    source="HyperLEDA (ledacat.cgi)",
                    method="HI linewidth proxy (0.35×W50)",
                    details={"leda_obj": leda_obj, "field": "w50", "w50": float(leda_w50)},
                )

        chosen = choose_best_candidate(cands)

        if args.verbose and (int(idx) % int(args.verbose_every) == 0):
            parts = []
            parts.append(f"[{int(idx)+1}/{n_hosts}] {galaxy}")
            parts.append(f"PGC={pgc if pgc is not None else '—'}")
            if log2vm is not None and np.isfinite(log2vm):
                parts.append(f"log(2Vm)={float(log2vm):.3f}")

            found_keys = ",".join(sorted(list(cands.keys()))) if cands else "none"
            parts.append(f"candidates={found_keys}")

            if chosen is None or not _is_finite_pos(chosen.value_kms):
                parts.append("CHOSEN=none")
            else:
                parts.append(f"CHOSEN={chosen.source} | {chosen.method} | σ={float(chosen.value_kms):.1f}")
                if chosen.error_kms is not None and np.isfinite(chosen.error_kms) and chosen.error_kms > 0:
                    parts.append(f"±{float(chosen.error_kms):.1f}")

            dt = time.time() - t0
            parts.append(f"t={dt:.2f}s")
            print_status(" | ".join(parts), "INFO")

        # Diagnostics: disagreements among direct measures
        direct_keys = [k for k in ["ho2009", "sdss", "leda_vdis"] if k in cands]
        direct_vals = [cands[k].value_kms for k in direct_keys if _is_finite_pos(cands[k].value_kms)]
        direct_spread = None
        if len(direct_vals) >= 2:
            direct_spread = float(np.nanmax(direct_vals) - np.nanmin(direct_vals))

        rows.append(
            {
                "galaxy": galaxy,
                "sigma_kms": float(chosen.value_kms) if chosen is not None and chosen.value_kms is not None else np.nan,
                "error_kms": float(chosen.error_kms) if chosen is not None and chosen.error_kms is not None else np.nan,
                "source": chosen.source if chosen is not None else "",
                "method": chosen.method if chosen is not None else "",
                "notes": json.dumps({k: asdict(v) for k, v in cands.items()}, sort_keys=True),
                "pgc": int(pgc) if pgc is not None else np.nan,
                "direct_sigma_spread_kms": direct_spread if direct_spread is not None else np.nan,
            }
        )

        diagnostics.append(
            {
                "galaxy": galaxy,
                "pgc": int(pgc) if pgc is not None else None,
                "chosen_key": None
                if chosen is None
                else next((k for k, v in cands.items() if v is chosen), None),
                "n_candidates": len(cands),
                "direct_keys": direct_keys,
                "direct_spread_kms": direct_spread,
            }
        )

        time.sleep(float(args.sleep))

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("galaxy")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    pkg_versions: Dict[str, Optional[str]] = {}
    for mod_name in ["numpy", "pandas", "astropy", "astroquery", "requests", "scipy"]:
        try:
            m = __import__(mod_name)
            pkg_versions[mod_name] = getattr(m, "__version__", None)
        except Exception:
            pkg_versions[mod_name] = None

    report = {
        "provenance": {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": " ".join([str(x) for x in sys.argv]),
            "python": {
                "executable": sys.executable,
                "version": sys.version,
            },
            "platform": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "packages": pkg_versions,
        },
        "inputs": {
            "hosts_csv": str(hosts_path),
            "n_hosts_raw": int(n_hosts_raw),
            "n_hosts_deduped": int(n_hosts_deduped),
            "hi_proxy_mode": args.hi_proxy,
            "use_ledacat": bool(args.use_ledacat),
            "leda_url": LEDA_URL if args.use_ledacat else None,
            "use_6dfgs": bool(args.use_6dfgs),
            "use_ho2007": bool(args.use_ho2007),
            "use_bassdr2": bool(args.use_bassdr2),
            "use_apj929": bool(args.use_apj929),
            "use_mnras482": bool(args.use_mnras482),
            "use_lit_overrides": bool(args.use_lit_overrides),
            "lit_overrides_csv": str(args.lit_overrides_csv) if args.use_lit_overrides else None,
            "sleep_seconds": float(args.sleep),
        },
        "counts": {
            "n_hosts": int(len(hosts)),
            "n_with_sigma": int(out_df["sigma_kms"].notna().sum()),
            "n_missing_sigma": int(out_df["sigma_kms"].isna().sum()),
        },
        "notes": {
            "priority_order": [
                "Literature override (manual)",
                "Ho+2009",
                "SDSS",
                "MNRAS 482:1427",
                "ApJ 929:84",
                "BASS DR2",
                "Ho 2007",
                "HyperLEDA vdis",
                "6dFGSv FP",
                "HyperLEDA HI log(2Vm)",
                "HyperLEDA vrot/1.7",
                "HyperLEDA w50 proxy",
                "HIPASS W50",
            ],
            "hi_proxy_definitions": {
                "kh13_w50": "sigma = 0.35 * W50, with W50 ~ 10**log(2Vm)",
                "calibrated_vmax": "sigma = 0.467 * Vmax + 42.931, with Vmax = 0.5 * 10**log(2Vm)",
            },
        },
        "diagnostics": diagnostics,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    print_status(f"Wrote regenerated sigma CSV: {out_csv}", "SUCCESS")
    print_status(f"Wrote regeneration report: {report_json}", "SUCCESS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
