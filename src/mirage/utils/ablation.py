#!/usr/bin/env python3
"""
Unified Ablation Study Runner

Runs the full MiRAGE pipeline for:
1. Baseline (all features enabled)
2. Each ablation configuration (one at a time)

Results are saved to separate directories under the base output path.

Usage:
    python run_ablation_study.py [--config config.yaml] [--skip-baseline] [--only ABLATION_NAME]
    
Examples:
    python run_ablation_study.py                          # Run all (baseline + all ablations)
    python run_ablation_study.py --skip-baseline          # Run only ablations
    python run_ablation_study.py --only disable_verifier  # Run only one ablation
    python run_ablation_study.py --only baseline          # Run only baseline
"""

import os
import sys
import yaml
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ablation configurations to test
ABLATION_MODES = [
    {
        "name": "baseline",
        "description": "Full MiRAGE framework (all features enabled)",
        "config_changes": {}  # No changes for baseline
    },
    {
        "name": "no_multihop",
        "description": "Disable Multihop Context Optimization Loop",
        "config_changes": {
            "ablation.disable_multihop_context.enabled": True
        }
    },
    {
        "name": "no_verifier", 
        "description": "Disable Verifier Agent",
        "config_changes": {
            "ablation.disable_verifier.enabled": True
        }
    },
    {
        "name": "no_persona",
        "description": "Disable Domain/Persona Injection",
        "config_changes": {
            "ablation.disable_persona.enabled": True
        }
    },
    {
        "name": "fixed_chunking",
        "description": "Use Fixed-Length Chunking (2048 tokens)",
        "config_changes": {
            "ablation.fixed_chunking.enabled": True
        }
    },
    {
        "name": "description_only",
        "description": "Multimodal: Description Only (no raw images)",
        "config_changes": {
            "ablation.description_only.enabled": True
        }
    },
    {
        "name": "image_only",
        "description": "Multimodal: Image Only (no generated descriptions)",
        "config_changes": {
            "ablation.image_only.enabled": True
        }
    },
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def set_nested_value(config: Dict, key_path: str, value: Any):
    """Set a nested dictionary value using dot notation."""
    keys = key_path.split('.')
    d = config
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def reset_all_ablations(config: Dict) -> Dict:
    """Reset all ablation settings to disabled."""
    if 'ablation' not in config:
        config['ablation'] = {}
    
    ablation = config['ablation']
    
    # Reset each ablation to disabled
    for ablation_key in ['disable_multihop_context', 'disable_verifier', 
                          'disable_persona', 'fixed_chunking', 
                          'description_only', 'image_only']:
        if ablation_key not in ablation:
            ablation[ablation_key] = {}
        ablation[ablation_key]['enabled'] = False
    
    return config


def apply_ablation_config(config: Dict, ablation_mode: Dict) -> Dict:
    """Apply ablation-specific configuration changes."""
    # First reset all ablations
    config = reset_all_ablations(config)
    
    # Apply specific changes for this ablation
    for key_path, value in ablation_mode.get('config_changes', {}).items():
        set_nested_value(config, key_path, value)
    
    return config


def get_output_dir(base_output_dir: str, ablation_name: str) -> str:
    """Generate output directory path for an ablation run."""
    return os.path.join(base_output_dir, ablation_name)


def run_pipeline(config_path: str, ablation_name: str) -> bool:
    """Run the main pipeline and return success status."""
    print(f"\n{'='*70}")
    print(f"🚀 Running: {ablation_name}")
    print(f"{'='*70}\n")
    
    try:
        # Run main.py
        result = subprocess.run(
            [sys.executable, 'main.py'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {ablation_name}: COMPLETED SUCCESSFULLY")
            return True
        else:
            print(f"\n❌ {ablation_name}: FAILED (exit code {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ {ablation_name}: ERROR - {e}")
        return False


def create_summary_report(results: List[Dict], output_dir: str):
    """Create a summary report of all ablation runs."""
    report_path = os.path.join(output_dir, "ablation_study_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ABLATION STUDY SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        f.write(f"Total Runs: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {total - successful}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("INDIVIDUAL RESULTS\n")
        f.write("-" * 70 + "\n\n")
        
        for result in results:
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            f.write(f"{result['name']:20} {status}\n")
            f.write(f"  Description: {result['description']}\n")
            f.write(f"  Output Dir:  {result['output_dir']}\n")
            f.write(f"  Duration:    {result.get('duration', 'N/A')}\n\n")
    
    print(f"\n📄 Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run unified ablation study for MiRAGE pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ablation_study.py                          # Run all
  python run_ablation_study.py --skip-baseline          # Skip baseline
  python run_ablation_study.py --only no_verifier       # Run specific ablation
  python run_ablation_study.py --only baseline          # Run baseline only
  python run_ablation_study.py --list                   # List available modes
        """
    )
    parser.add_argument('--config', default='config.yaml', 
                        help='Path to config file (default: config.yaml)')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip the baseline run')
    parser.add_argument('--only', type=str, default=None,
                        help='Run only a specific ablation mode')
    parser.add_argument('--list', action='store_true',
                        help='List available ablation modes and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # List available modes
    if args.list:
        print("\nAvailable Ablation Modes:")
        print("-" * 50)
        for mode in ABLATION_MODES:
            print(f"  {mode['name']:20} - {mode['description']}")
        print()
        return 0
    
    # Validate --only argument
    if args.only:
        valid_names = [m['name'] for m in ABLATION_MODES]
        if args.only not in valid_names:
            print(f"❌ Error: Unknown ablation mode '{args.only}'")
            print(f"   Valid modes: {', '.join(valid_names)}")
            return 1
    
    # Load original config
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found: {config_path}")
        return 1
    
    print("=" * 70)
    print("🔬 UNIFIED ABLATION STUDY")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and backup original config
    original_config = load_config(config_path)
    backup_path = config_path + '.backup'
    shutil.copy2(config_path, backup_path)
    print(f"Config backed up: {backup_path}")
    
    # Get base output directory
    base_output_dir = original_config.get('paths', {}).get('output_dir', 'output')
    print(f"Base Output Dir: {base_output_dir}")
    
    # Determine which modes to run
    modes_to_run = ABLATION_MODES.copy()
    
    if args.only:
        modes_to_run = [m for m in modes_to_run if m['name'] == args.only]
    elif args.skip_baseline:
        modes_to_run = [m for m in modes_to_run if m['name'] != 'baseline']
    
    print(f"\nModes to run: {[m['name'] for m in modes_to_run]}")
    print("-" * 70)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Would execute the following:")
        for mode in modes_to_run:
            output_dir = get_output_dir(base_output_dir, mode['name'])
            print(f"\n  [{mode['name']}]")
            print(f"    Description: {mode['description']}")
            print(f"    Output: {output_dir}")
            print(f"    Config changes: {mode['config_changes']}")
        print("\n✅ Dry run complete. Use without --dry-run to execute.")
        return 0
    
    # Run each ablation mode
    results = []
    
    try:
        for mode in modes_to_run:
            start_time = datetime.now()
            
            # Create output directory for this ablation
            output_dir = get_output_dir(base_output_dir, mode['name'])
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare config for this run
            config = load_config(config_path)  # Reload fresh each time
            config = apply_ablation_config(config, mode)
            
            # Update output directory
            if 'paths' not in config:
                config['paths'] = {}
            config['paths']['output_dir'] = output_dir
            
            # Save modified config
            save_config(config, config_path)
            
            print(f"\n📁 Output: {output_dir}")
            print(f"📝 Mode: {mode['description']}")
            
            # Run the pipeline
            success = run_pipeline(config_path, mode['name'])
            
            end_time = datetime.now()
            duration = str(end_time - start_time).split('.')[0]  # Remove microseconds
            
            results.append({
                'name': mode['name'],
                'description': mode['description'],
                'output_dir': output_dir,
                'success': success,
                'duration': duration
            })
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Ablation study interrupted by user")
        
    finally:
        # Restore original config
        print(f"\n{'='*70}")
        print("Restoring original configuration...")
        shutil.copy2(backup_path, config_path)
        os.remove(backup_path)
        print("✅ Original config restored")
    
    # Generate summary report
    if results:
        create_summary_report(results, base_output_dir)
        
        # Print final summary
        print(f"\n{'='*70}")
        print("ABLATION STUDY COMPLETE")
        print(f"{'='*70}")
        
        successful = sum(1 for r in results if r['success'])
        print(f"\nResults: {successful}/{len(results)} successful")
        
        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"  {status} {r['name']:20} ({r['duration']})")
        
        print(f"\nResults saved in: {base_output_dir}/")
        print(f"Summary report: {base_output_dir}/ablation_study_summary.txt")
    
    return 0 if all(r['success'] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

