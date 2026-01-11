#!/usr/bin/env python3
"""Unified PDF Processing Script
Compresses PDF and embeds comprehensive metadata in one operation.

This script processes the TEP-H0 manuscript PDF (Paper 12: "The Cepheid Bias: Resolving
the Hubble Tension") by compressing it for web distribution and embedding complete
academic metadata for proper indexing and citation.

Usage:
    python process_pdf.py <input_pdf> [--quality ebook|printer|prepress|default]
    
Example:
    python process_pdf.py site/public/docs/Smawfield_2026_TEP-H0_v0.2_KingstonUponHull.pdf --quality ebook
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import tempfile


def compress_pdf(input_path, output_path, quality='ebook'):
    """Compress PDF using Ghostscript."""
    quality_settings = {
        'screen': '/screen',      # 72 dpi
        'ebook': '/ebook',        # 150 dpi
        'printer': '/printer',    # 300 dpi
        'prepress': '/prepress',  # 300 dpi, color preserving
        'default': '/default'
    }
    
    if quality not in quality_settings:
        raise ValueError(f"Quality must be one of: {', '.join(quality_settings.keys())}")
    
    gs_quality = quality_settings[quality]
    
    # Get original size
    original_size = os.path.getsize(input_path)
    
    # Compress using Ghostscript
    cmd = [
        'gs',
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        f'-dPDFSETTINGS={gs_quality}',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        f'-sOutputFile={output_path}',
        input_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        compressed_size = os.path.getsize(output_path)
        reduction = ((original_size - compressed_size) / original_size) * 100
        
        return {
            'original_mb': original_size / (1024 * 1024),
            'compressed_mb': compressed_size / (1024 * 1024),
            'reduction_pct': reduction
        }
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ghostscript compression failed: {e.stderr.decode()}")


def embed_metadata(pdf_path, metadata):
    """Embed metadata into PDF using exiftool."""
    cmd = ['exiftool']
    
    # Add all metadata fields
    for key, value in metadata.items():
        cmd.extend([f'-{key}={value}'])
    
    # Overwrite original
    cmd.extend(['-overwrite_original', pdf_path])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Exiftool metadata embedding failed: {e.stderr.decode()}")


def verify_metadata(pdf_path, expected_fields):
    """Verify metadata was embedded correctly."""
    cmd = ['exiftool'] + [f'-{field}' for field in expected_fields] + [pdf_path]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Compress PDF and embed metadata in one operation'
    )
    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument(
        '--quality',
        choices=['screen', 'ebook', 'printer', 'prepress', 'default'],
        default='ebook',
        help='Compression quality (default: ebook)'
    )
    parser.add_argument(
        '--doi',
        default='10.5281/zenodo.18209703',
        help='DOI to embed in metadata'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_pdf).resolve()
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing: {input_path}")
    print(f"Quality: {args.quality}")
    print()
    
    # Step 1: Compress PDF
    print("Step 1: Compressing PDF...")
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        stats = compress_pdf(str(input_path), tmp_path, args.quality)
        
        # Replace original with compressed version
        os.replace(tmp_path, str(input_path))
        
        print(f"  Original:    {stats['original_mb']:.2f} MB")
        print(f"  Compressed:  {stats['compressed_mb']:.2f} MB")
        print(f"  Reduction:   {stats['reduction_pct']:.1f}%")
        print()
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"Error during compression: {e}")
        sys.exit(1)
    
    # Step 2: Embed metadata
    print("Step 2: Embedding metadata...")
    
    # Paper metadata - must match manuscript, CITATION.cff, and zenodo.txt
    metadata = {
        # Core identification
        'Title': 'The Cepheid Bias: Resolving the Hubble Tension',
        'Author': 'Matthew Lukin Smawfield',
        'Creator': 'Matthew Lukin Smawfield',
        
        # Scientific abstract with key results
        'Subject': (
            'The Hubble Tension—the persistent 5σ discrepancy between local distance-ladder measurements '
            '(H0 ≈ 73 km/s/Mpc) and early-universe CMB inference (H0 = 67.4 ± 0.5 km/s/Mpc)—represents '
            'a significant challenge in precision cosmology. This study proposes that the tension arises '
            'from a systematic, environment-dependent bias in Cepheid-based distances, as predicted by '
            'the Temporal Equivalence Principle (TEP). '
            'Analysis of the SH0ES Cepheid sample (N=29), stratified by host galaxy velocity dispersion, '
            'reveals a strong correlation (Spearman ρ = 0.638, p = 0.0002) between host potential depth '
            'and derived H0. Low-density hosts (σ ≤ 88 km/s) yield H0 = 65.44 ± 1.88 km/s/Mpc, while '
            'high-density hosts (σ > 88 km/s) yield H0 = 75.01 ± 1.23 km/s/Mpc—a systematic offset of '
            'ΔH0 = 9.57 km/s/Mpc that exceeds the Hubble Tension magnitude. '
            'Application of the TEP conformal correction with optimal coupling α = 0.87 ± 0.24 and '
            'effective calibrator reference σ_ref = 75.25 km/s yields a unified local Hubble constant '
            'of H0 = 67.27 ± 1.54 km/s/Mpc, reducing the tension with Planck to 0.08σ. '
            'Keywords: Hubble tension, Cepheid variables, distance ladder, velocity dispersion, '
            'temporal equivalence principle, gravitational time dilation.'
        ),
        
        # Keywords for indexing
        'Keywords': (
            'Hubble Tension; Hubble Constant; H0; Cepheid Variables; Period-Luminosity Relation; '
            'Distance Ladder; Type Ia Supernovae; SH0ES; Planck; CMB; Velocity Dispersion; '
            'Gravitational Potential; Temporal Equivalence Principle; TEP; Time Dilation; '
            'Environmental Bias; Screening; Modified Gravity; Cosmology'
        ),
        
        # Production metadata
        'Producer': 'TEP-H0 Research Project (Paper 12) - Version 0.2 (Kingston upon Hull)',
        
        # Rights and identifiers
        'Copyright': 'Creative Commons Attribution 4.0 International License (CC BY 4.0)',
        
        # Dates
        'CreationDate': '2026:01:11 00:00:00',
        'ModifyDate': '2026:01:11 00:00:00',
        
        # XMP Dublin Core metadata (exiftool uses these prefixes)
        'XMP-dc:Creator': 'Matthew Lukin Smawfield',
        'XMP-dc:Title': 'The Cepheid Bias: Resolving the Hubble Tension',
        'XMP-dc:Description': 'Resolving the Hubble Tension via environment-dependent Cepheid calibration',
        'XMP-dc:Rights': 'CC BY 4.0',
        'XMP-dc:Identifier': f'doi:{args.doi}',
        'XMP-dc:Source': 'https://matthewsmawfield.github.io/TEP-H0/',
        'XMP-dc:Publisher': 'Zenodo',
        'XMP-dc:Date': '2026-01-11',
        'XMP-dc:Type': 'Preprint',
        'XMP-dc:Format': 'application/pdf',
        'XMP-dc:Language': 'en',
        
        # PRISM (Publishing Requirements for Industry Standard Metadata)
        'XMP-prism:DOI': args.doi,
        'XMP-prism:URL': 'https://matthewsmawfield.github.io/TEP-H0/',
        'XMP-prism:VersionIdentifier': '0.2',
        'XMP-prism:PublicationName': 'TEP Research Series',
        
        # PDF/A metadata
        'XMP-pdfaid:Part': '1',
        'XMP-pdfaid:Conformance': 'B'
    }
    
    try:
        embed_metadata(str(input_path), metadata)
        print("  Metadata embedded successfully")
        print()
        
    except Exception as e:
        print(f"Error during metadata embedding: {e}")
        sys.exit(1)
    
    # Step 3: Verify
    print("Step 3: Verifying metadata...")
    verification = verify_metadata(
        str(input_path),
        ['Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Copyright']
    )
    
    if verification:
        print("  ✓ Metadata verified")
        print()
        print("Verification output:")
        print(verification)
    else:
        print("  ⚠ Could not verify metadata")
    
    print()
    print(f"✓ Processing complete: {input_path}")
    print(f"  Final size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")


if __name__ == '__main__':
    main()
