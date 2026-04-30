"""
Diagnostic script to verify ELECTRA model path and files.
"""

import os
import sys

print("="*80)
print("🔍 ELECTRA MODEL PATH DIAGNOSTICS")
print("="*80)

# Get base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\n✓ Base directory: {base_dir}")

# Check nested path (primary)
nested_path = os.path.join(
    base_dir, 
    "Crisis Detection Model", 
    "electra_suicidal_text_detector",
    "electra_suicidal_text_detector"
)
print(f"\n📂 Checking nested path:")
print(f"   {nested_path}")
print(f"   Exists: {os.path.exists(nested_path)}")

if os.path.exists(nested_path):
    print(f"   Files:")
    for file in os.listdir(nested_path):
        file_path = os.path.join(nested_path, file)
        size = os.path.getsize(file_path) if os.path.isfile(file_path) else "DIR"
        print(f"     - {file} ({size})")

# Check flat path (fallback)
flat_path = os.path.join(
    base_dir, 
    "Crisis Detection Model", 
    "electra_suicidal_text_detector"
)
print(f"\n📂 Checking flat path (fallback):")
print(f"   {flat_path}")
print(f"   Exists: {os.path.exists(flat_path)}")

if os.path.exists(flat_path):
    print(f"   Contents:")
    for item in os.listdir(flat_path):
        item_path = os.path.join(flat_path, item)
        if os.path.isdir(item_path):
            print(f"     📁 {item}/")
        else:
            print(f"     📄 {item}")

# Check for required model files
print(f"\n✅ Required model files:")
required_files = ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"]
for required_file in required_files:
    nested_file = os.path.join(nested_path, required_file)
    flat_file = os.path.join(flat_path, required_file)
    
    nested_exists = os.path.exists(nested_file)
    flat_exists = os.path.exists(flat_file)
    
    status = "✓ FOUND (nested)" if nested_exists else ("✓ FOUND (flat)" if flat_exists else "✗ MISSING")
    print(f"   {required_file:30} {status}")

print("\n" + "="*80)
