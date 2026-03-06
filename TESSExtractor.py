#!/usr/bin/env python3
"""CLI to extract and correct TESS light curves with CBVs."""

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.timeseries import LombScargle
import astropy.units as u
from astroquery.mast import Catalogs
from astroquery.mast import TesscutClass
from astroquery.simbad import Simbad
from scipy.interpolate import interp1d

import photometry


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def build_output_paths(output_png: str):
    png_path = Path(output_png)
    if not png_path.parent or str(png_path.parent) == ".":
        png_path = Path.cwd() / png_path
    stem = png_path.with_suffix("")
    combined_csv = Path(f"{stem}_lc.csv")
    return png_path, combined_csv


def save_lightcurve_csv(path: Path, time, flux_raw, flux_corrected, flux_err):
    data = np.column_stack((time, flux_raw, flux_corrected, flux_err))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="time,flux_raw,flux_corrected,flux_err",
        comments="",
    )


def compute_lomb_scargle(time, flux_corrected):
    finite = np.isfinite(time) & np.isfinite(flux_corrected)
    if np.count_nonzero(finite) < 10:
        return None

    t = np.asarray(time[finite], dtype=float)
    y = np.asarray(flux_corrected[finite], dtype=float)

    baseline = float(np.ptp(t))
    if not np.isfinite(baseline) or baseline <= 0:
        return None

    dt = np.diff(np.sort(t))
    dt = dt[dt > 0]
    if dt.size == 0:
        return None

    min_freq = 1.0 / baseline
    max_freq = 0.5 / float(np.median(dt))
    if not np.isfinite(max_freq) or max_freq <= min_freq:
        return None

    ls = LombScargle(t, y, center_data=True, fit_mean=True)
    freq, power = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=8,
    )
    if freq.size == 0:
        return None

    best_idx = int(np.argmax(power))
    best_period = 1.0 / float(freq[best_idx])
    return t, y, freq, power, best_period


def resolve_target_with_simbad(target: str):
    simbad = Simbad()
    simbad.add_votable_fields("ra", "dec")
    result = simbad.query_object(target)
    if result is None or len(result) == 0:
        raise ValueError(f"Could not resolve '{target}' in SIMBAD.")

    colnames = set(result.colnames)

    if "RA_d" in colnames and "DEC_d" in colnames:
        return SkyCoord(float(result[0]["RA_d"]) * u.deg, float(result[0]["DEC_d"]) * u.deg, frame="icrs")
    if "ra_d" in colnames and "dec_d" in colnames:
        return SkyCoord(float(result[0]["ra_d"]) * u.deg, float(result[0]["dec_d"]) * u.deg, frame="icrs")

    if "RA" in colnames and "DEC" in colnames:
        ra_str = result[0]["RA"]
        dec_str = result[0]["DEC"]
        return SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg), frame="icrs")
    if "ra" in colnames and "dec" in colnames:
        ra_str = result[0]["ra"]
        dec_str = result[0]["dec"]
        return SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg), frame="icrs")

    raise ValueError(
        "SIMBAD resolved the target but did not return coordinates in supported columns "
        f"(received columns: {result.colnames})."
    )


def query_tic_properties(coord: SkyCoord):
    tic = Catalogs.query_region(coord, radius=0.02 * u.deg, catalog="TIC")
    if len(tic) == 0:
        raise ValueError("No TIC counterpart found near the SIMBAD position.")

    coord_deg = np.array([coord.ra.deg, coord.dec.deg])
    d2 = (tic["ra"] - coord_deg[0]) ** 2 + (tic["dec"] - coord_deg[1]) ** 2
    idx = int(np.argmin(d2))

    return {
        "tic_id": int(tic[idx]["ID"]),
        "ra": float(tic[idx]["ra"]),
        "dec": float(tic[idx]["dec"]),
        "tmag": float(tic[idx]["Tmag"]),
    }


def choose_aperture_radii(tmag: float):
    if tmag > 13:
        return 1.5, 2.5, 3.5
    if 11 < tmag <= 13:
        return 2.0, 3.0, 4.0
    if 9 < tmag <= 11:
        return 2.5, 3.5, 4.5
    return 3.0, 4.0, 5.0


def _try_world_to_pix(wcs_obj, ra_deg: float, dec_deg: float):
    px = np.asarray(wcs_obj.all_world2pix([[ra_deg, dec_deg]], 0), dtype=float)
    if px.shape == (1, 2) and np.all(np.isfinite(px)):
        return px
    return None


def compute_starloc_with_fallback(wcs_obj, hdu, simbad_coord: SkyCoord, tic: dict):
    starloc = _try_world_to_pix(wcs_obj, simbad_coord.ra.deg, simbad_coord.dec.deg)
    if starloc is not None:
        return starloc, None

    starloc = _try_world_to_pix(wcs_obj, tic["ra"], tic["dec"])
    if starloc is not None:
        return starloc, "Could not project SIMBAD coordinates with WCS; TIC position was used."

    ny, nx = hdu[1].data["FLUX"][0].shape
    center = np.array([[nx / 2.0, ny / 2.0]], dtype=float)
    return center, "Could not project SIMBAD/TIC coordinates with WCS; cutout center was used."


def find_cbv_url(sector: int, camera: int, ccd: int, master_file: Path):
    key = f"s{sector:04d}-{camera}-{ccd}"
    with master_file.open("r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if key in url:
                return url
    raise FileNotFoundError(f"CBV not found for sector={sector}, camera={camera}, ccd={ccd} in {master_file}.")


def get_cbv_local_file(sector: int, camera: int, ccd: int, master_file: Path, cache_dir: Path):
    cbv_url = find_cbv_url(sector, camera, ccd, master_file)
    cache_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(cbv_url)
    filename = Path(parsed.path).name
    if not filename:
        filename = f"cbv_s{sector:04d}_{camera}_{ccd}.fits"
    local_path = cache_dir / filename

    from_cache = local_path.exists() and local_path.stat().st_size > 0
    if not from_cache:
        urlretrieve(cbv_url, local_path)

    return cbv_url, local_path, from_cache


def load_cbv_vectors(cbv_url: str):
    with fits.open(cbv_url) as hdu:
        data = hdu[1].data
        time_cbv = np.asarray(data["TIME"], dtype=float)
        vector_cols = [c for c in data.columns.names if c.startswith("VECTOR_")]
        vector_cols = sorted(vector_cols, key=lambda x: int(x.split("_")[1]))
        vectors = [np.asarray(data[col], dtype=float) for col in vector_cols]

    if not vectors:
        raise ValueError("CBV file does not contain VECTOR_N columns.")

    return time_cbv, vectors


def robust_lstsq_model(X, y, max_iter=7, sigma_clip=5.0):
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if finite.sum() <= X.shape[1]:
        raise ValueError("Not enough valid points to fit CBVs.")

    mask = finite.copy()
    for _ in range(max_iter):
        coeffs, _, _, _ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        model = X @ coeffs
        resid = y - model

        med = np.nanmedian(resid[mask])
        mad = np.nanmedian(np.abs(resid[mask] - med))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma == 0:
            break

        new_mask = finite & (np.abs(resid - med) < sigma_clip * sigma)
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    coeffs, _, _, _ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    model = X @ coeffs
    return coeffs, model


def robust_cbv_fit(time, flux, time_cbv, cbv_vectors, n_vectors, max_iter=7, sigma_clip=5.0):
    n_vectors = min(n_vectors, len(cbv_vectors))
    interp_vectors = [interp1d(time_cbv, v, bounds_error=False, fill_value="extrapolate")(time) for v in cbv_vectors[:n_vectors]]
    X = np.vstack(interp_vectors).T
    y = np.asarray(flux, dtype=float)
    coeffs, model = robust_lstsq_model(X, y, max_iter=max_iter, sigma_clip=sigma_clip)
    corrected = (y - model) + 1.0
    return coeffs, model, corrected, n_vectors


def parse_cbv_layer_names(layers_text: str):
    layers = [p.strip() for p in layers_text.split(",") if p.strip()]
    if not layers:
        raise ValueError("`--cbv_layers` cannot be empty.")
    return layers


def parse_cbv_layer_vector_counts(counts_text: str):
    counts = [int(p.strip()) for p in counts_text.split(",") if p.strip()]
    if not counts:
        raise ValueError("`--cbv_vectors_per_layer` cannot be empty.")
    if any(n < 1 for n in counts):
        raise ValueError("All values in `--cbv_vectors_per_layer` must be >= 1.")
    return counts


def normalize_layer_label(layer_label: str):
    s = layer_label.strip().upper().replace(" ", "")
    if s in {"SPIKE"}:
        return "SPIKE"
    if s in {"SINGLESCALE", "SINGLESCALE"}:
        return "SINGLESCALE"
    if s.startswith("MULTISCALE."):
        band = s.split(".", 1)[1]
        if band in {"1", "2", "3"}:
            return f"MULTISCALE.{band}"
    raise ValueError(
        f"Unsupported CBV layer: '{layer_label}'. Use MultiScale.1, MultiScale.2, MultiScale.3, Spike, or SingleScale."
    )


def _has_time_and_vectors(hdu):
    if not hasattr(hdu, "columns") or hdu.columns is None:
        return False
    names = set(hdu.columns.names or [])
    return ("TIME" in names) and any(n.startswith("VECTOR_") for n in names)


def _extract_descriptor(hdu):
    extname = str(hdu.header.get("EXTNAME", "")).upper().replace(" ", "")
    htxt = " ".join(
        [
            str(hdu.header.get("CBV_TYPE", "")),
            str(hdu.header.get("CBVTYPE", "")),
            str(hdu.header.get("TYPE", "")),
            str(hdu.header.get("BAND", "")),
            str(hdu.header.get("BANDNUM", "")),
        ]
    ).upper().replace(" ", "")
    return f"{extname} {htxt}"


def _layer_matches_hdu(layer_norm: str, hdu):
    desc = _extract_descriptor(hdu)
    if layer_norm == "SPIKE":
        return "SPIKE" in desc
    if layer_norm == "SINGLESCALE":
        return ("SINGLESCALE" in desc) or ("SINGLESCALE" in desc) or ("-S_CBV" in desc) or ("_S_CBV" in desc)
    if layer_norm.startswith("MULTISCALE."):
        band = layer_norm.split(".", 1)[1]
        if "MULTISCALE" not in desc:
            return False
        return any(tok in desc for tok in [f".{band}", f"_{band}", f"-{band}", f"BAND{band}"])
    return False


def _find_hdu_for_layer(hdul, layer_norm: str):
    candidates = [h for h in hdul if _has_time_and_vectors(h)]
    for h in candidates:
        if _layer_matches_hdu(layer_norm, h):
            return h

    if layer_norm.startswith("MULTISCALE."):
        band = int(layer_norm.split(".", 1)[1])
        ms_candidates = [h for h in candidates if "MULTISCALE" in _extract_descriptor(h)]
        if len(ms_candidates) >= band:
            return ms_candidates[band - 1]

    if layer_norm == "SINGLESCALE" and candidates:
        return candidates[0]

    available = [str(h.header.get("EXTNAME", f"HDU{idx}")) for idx, h in enumerate(candidates)]
    raise ValueError(f"Layer '{layer_norm}' was not found in CBV FITS. Available extensions: {available}")


def _build_design_matrix_from_hdu(hdu, time, requested):
    data = hdu.data
    time_cbv = np.asarray(data["TIME"], dtype=float)
    vector_cols = [c for c in hdu.columns.names if c.startswith("VECTOR_")]
    vector_cols = sorted(vector_cols, key=lambda x: int(x.split("_")[1]))
    n_use = min(requested, len(vector_cols))
    if n_use < 1:
        raise ValueError(f"Extension {hdu.header.get('EXTNAME', '')} has no CBV vectors.")
    interp_vectors = [interp1d(time_cbv, np.asarray(data[col], dtype=float), bounds_error=False, fill_value="extrapolate")(time) for col in vector_cols[:n_use]]
    X = np.column_stack(interp_vectors)
    return X, n_use


def robust_cbv_fit_by_types(time, flux, cbv_fits_path, layer_names, layer_counts, max_iter=7, sigma_clip=5.0):
    if len(layer_names) != len(layer_counts):
        raise ValueError("The number of layers and vectors-per-layer entries must match.")

    layer_norms = [normalize_layer_label(name) for name in layer_names]
    X_blocks = []
    per_layer_info = []
    with fits.open(cbv_fits_path) as hdul:
        for layer_name, layer_norm, requested in zip(layer_names, layer_norms, layer_counts):
            hdu = _find_hdu_for_layer(hdul, layer_norm)
            X_layer, n_use = _build_design_matrix_from_hdu(hdu, time, requested)
            X_blocks.append(X_layer)
            per_layer_info.append(
                {
                    "layer_name": layer_name,
                    "layer_norm": layer_norm,
                    "requested": requested,
                    "used": n_use,
                    "extname": str(hdu.header.get("EXTNAME", "")),
                }
            )

    if not X_blocks:
        raise ValueError("No CBVs could be loaded for the requested layers.")

    X = np.hstack(X_blocks)
    y = np.asarray(flux, dtype=float)
    coeffs, model = robust_lstsq_model(X, y, max_iter=max_iter, sigma_clip=sigma_clip)
    corrected = (y - model) + 1.0
    return coeffs, model, corrected, per_layer_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a TESS light curve for a SIMBAD target and correct systematics with CBVs."
    )
    parser.add_argument("--target", required=True, help="SIMBAD target name (e.g. 'LkCa 5').")
    parser.add_argument(
        "--num_cbv_vectors",
        "--vectors_CBV",
        "--n_vectors_CBV",
        type=int,
        default=None,
        help="Number of CBV vectors to include (if not provided, prompt in console).",
    )
    parser.add_argument(
        "--cbv_layers",
        default=None,
        help="Comma-separated CBV layer names. Example: MultiScale.1,MultiScale.2,MultiScale.3,Spike",
    )
    parser.add_argument(
        "--cbv_vectors_per_layer",
        default=None,
        help="Comma-separated number of vectors per layer. Example: 4,4,4,8",
    )
    parser.add_argument("--sector", type=int, default=None, help="TESS sector to process (if omitted, prompt in console).")
    parser.add_argument("--cutout_size", type=int, default=10, help="TESScut cutout size in pixels.")
    parser.add_argument("--master_cbv", default="master_cbv.txt", help="Path to CBV URL master file.")
    parser.add_argument(
        "--cbv_cache_dir",
        default=".cbv_cache",
        help="Local cache directory for CBV FITS files.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path (auto-generated if omitted).")
    parser.add_argument("--show", action="store_true", help="Display plot in addition to saving it.")
    return parser.parse_args()


def main():
    args = parse_args()

    layer_names = None
    layer_counts = None
    if args.cbv_layers:
        layer_names = parse_cbv_layer_names(args.cbv_layers)
        if args.cbv_vectors_per_layer:
            layer_counts = parse_cbv_layer_vector_counts(args.cbv_vectors_per_layer)
        else:
            layer_counts = []
            for lname in layer_names:
                n = int(input(f"How many vectors for layer {lname}? ").strip())
                if n < 1:
                    raise ValueError("Each layer must have at least 1 vector.")
                layer_counts.append(n)
    else:
        n_vectors = args.num_cbv_vectors
        if n_vectors is None:
            n_vectors = int(input("How many CBV vectors should be included in the correction? ").strip())
        if n_vectors < 1:
            raise ValueError("The number of CBV vectors must be >= 1.")

    coord = resolve_target_with_simbad(args.target)
    tic = query_tic_properties(coord)

    tesscut = TesscutClass()
    sector_table = tesscut.get_sectors(objectname=args.target)
    if len(sector_table) == 0:
        raise ValueError(f"No TESS sectors available for '{args.target}'.")

    sectors = sorted([int(s) for s in sector_table["sector"]])
    if args.sector is not None:
        sector = args.sector
    else:
        print("Available sectors:", ", ".join(str(s) for s in sectors))
        sector = int(input("Which sector do you want to process? ").strip())
    if sector not in sectors:
        raise ValueError(f"Sector {sector} is not available for this target. Available: {sectors}")

    hdulist = tesscut.get_cutouts(objectname=args.target, size=args.cutout_size, sector=sector)
    hdu = hdulist[0]

    camera = int(hdu[0].header["CAMERA"])
    ccd = int(hdu[0].header["CCD"])

    wcs = photometry.WCS(hdu[2].header)
    starloc, starloc_warning = compute_starloc_with_fallback(wcs, hdu, coord, tic)

    rap, rin, rout = choose_aperture_radii(tic["tmag"])
    photometry.aperture_annulus(starloc, r_ap=rap, r_in=rin, r_out=rout)
    time, flux, flux_err = photometry.LC_flux(hdu[1])
    cbv_url, cbv_local_path, cbv_loaded_from_cache = get_cbv_local_file(
        sector=sector,
        camera=camera,
        ccd=ccd,
        master_file=Path(args.master_cbv),
        cache_dir=Path(args.cbv_cache_dir),
    )

    if layer_names is not None:
        coeffs, cbv_model, flux_corrected, layer_results = robust_cbv_fit_by_types(
            time=time,
            flux=flux,
            cbv_fits_path=cbv_local_path,
            layer_names=layer_names,
            layer_counts=layer_counts,
        )
        n_used = int(sum(item["used"] for item in layer_results))
        model_label = (
            "CBV model by layer ("
            + ",".join([f"{it['layer_name']}:{it['used']}" for it in layer_results])
            + ")"
        )
    else:
        time_cbv, cbv_vectors = load_cbv_vectors(str(cbv_local_path))
        coeffs, cbv_model, flux_corrected, n_used = robust_cbv_fit(
            time=time,
            flux=flux,
            time_cbv=time_cbv,
            cbv_vectors=cbv_vectors,
            n_vectors=n_vectors,
        )
        layer_results = []
        model_label = f"CBV model ({n_used} vectors)"

    raw_for_plot = flux + 1.0
    model_for_plot = cbv_model + 1.0

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.18)

    ax_raw = fig.add_subplot(gs[0, :])
    ax_model = fig.add_subplot(gs[1, :], sharex=ax_raw)
    ax_phase = fig.add_subplot(gs[2, 0])
    ax_periodogram = fig.add_subplot(gs[2, 1])

    ax_raw.errorbar(time, raw_for_plot, yerr=flux_err, fmt=".", ms=2, alpha=0.25, color="royalblue", label="Raw")
    ax_raw.plot(time, model_for_plot, color="orange", lw=1.0, label=model_label)
    ax_raw.set_ylabel("Raw + systematics")
    ax_raw.set_title(f"{args.target} | TIC {tic['tic_id']} | Sector {sector}")
    ax_raw.legend(loc="best")

    ax_model.errorbar(time, flux_corrected, yerr=flux_err, fmt=".", ms=2, alpha=0.35, color="seagreen")
    ax_model.set_ylabel("Corrected flux")
    ax_model.set_xlabel("Time (BJD - 2457000) [days]")

    ls_result = compute_lomb_scargle(time, flux_corrected)
    if ls_result is not None:
        t_valid, y_valid, freq, power, best_period = ls_result

        phase = ((t_valid - np.min(t_valid)) / best_period) % 1.0
        phase_plot = np.concatenate([phase, phase + 1.0])
        flux_phase_plot = np.concatenate([y_valid, y_valid])
        ax_phase.plot(phase_plot, flux_phase_plot, ".", ms=2, alpha=0.35, color="teal")
        ax_phase.set_xlim(0.0, 2.0)
        ax_phase.set_xlabel("Phase")
        ax_phase.set_ylabel("Corrected flux")
        ax_phase.set_title(f"Phase-folded corrected light curve (P={best_period:.2f} d)")

        period = 1.0 / freq
        order = np.argsort(period)
        ax_periodogram.plot(period[order], power[order], color="purple", lw=1.0)
        ax_periodogram.axvline(best_period, color="crimson", lw=1.0, ls="--", label=f"Best P={best_period:.2f} d")
        ax_periodogram.set_xscale("log")
        ax_periodogram.set_xlabel("Period [days]")
        ax_periodogram.set_ylabel("Lomb-Scargle power")
        ax_periodogram.set_title("Lomb-Scargle periodogram")
        ax_periodogram.legend(loc="best")
    else:
        best_period = None
        ax_phase.text(0.5, 0.5, "Not enough data for phase fold", transform=ax_phase.transAxes, ha="center", va="center")
        ax_phase.set_xlabel("Phase")
        ax_phase.set_ylabel("Corrected flux")
        ax_phase.set_title("Phase-folded corrected light curve")

        ax_periodogram.text(
            0.5,
            0.5,
            "Not enough data for Lomb-Scargle",
            transform=ax_periodogram.transAxes,
            ha="center",
            va="center",
        )
        ax_periodogram.set_xlabel("Period [days]")
        ax_periodogram.set_ylabel("Lomb-Scargle power")
        ax_periodogram.set_title("Lomb-Scargle periodogram")

    for ax in [ax_raw, ax_model, ax_phase, ax_periodogram]:
        ax.grid(alpha=0.2)

    fig.tight_layout()

    out = args.output
    if out is None:
        out = f"{sanitize_name(args.target)}_s{sector}_cbv{n_used}.png"
    out_png, out_csv = build_output_paths(out)
    fig.savefig(out_png, dpi=180)

    save_lightcurve_csv(out_csv, time, raw_for_plot, flux_corrected, flux_err)

    print(f"Target: {args.target}")
    print(f"TIC ID: {tic['tic_id']} | RA: {tic['ra']:.6f} | DEC: {tic['dec']:.6f}")
    print(f"Sector: {sector} | Camera: {camera} | CCD: {ccd}")
    if starloc_warning:
        print(f"Warning: {starloc_warning}")
    print(f"CBV URL: {cbv_url}")
    print(f"CBV local: {cbv_local_path}")
    print(f"CBV loaded from cache: {'yes' if cbv_loaded_from_cache else 'no (downloaded in this run)'}")
    print(f"CBV vectors used: {n_used}")
    if best_period is not None:
        print(f"Best Lomb-Scargle period: {best_period:.2f} days")
    if layer_names is not None:
        print("Requested CBV layers:", layer_names)
        print("Requested vectors per layer:", layer_counts)
        k = 0
        for layer in layer_results:
            n = layer["used"]
            coeffs_layer = coeffs[k : k + n]
            coeffs_str = np.array2string(coeffs_layer, precision=6)
            print(
                f"Layer {layer['layer_name']} (EXTNAME={layer['extname']}): "
                f"requested={layer['requested']}, used={layer['used']} | Coefficients: {coeffs_str}"
            )
            k += n
    else:
        print("Coefficients:", np.array2string(coeffs, precision=6))
    print(f"Plot saved to: {out_png}")
    print(f"Light curve CSV (raw+corrected): {out_csv}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()


"""python codigo.py --target "Brun 691" --cbv_layers "MultiScale.1,MultiScale.2,MultiScale.3,Spike" --cbv_vectors_per_layer "4,4,4,8"""
