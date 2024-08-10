#!/bin/bash

# Function to download and extract files from HuggingFace
download_and_extract() {
    local repo_name=$1
    local filename=$2
    local url="https://huggingface.co/${repo_name}/resolve/main/${filename}?download=true"
    
    echo "Downloading ${filename}..."
    wget -O "${filename}" "${url}"
    
    echo "Extracting ${filename}..."
    unzip "${filename}"
    
    echo "Removing ${filename}..."
    rm "${filename}"
}

# Install requirements
pip install -r requirements.txt
pip install -e .
git submodule update --init

pip install nbstripout
pip install ipykernel
nbstripout --install

# Install necessary tools
apt-get update
apt-get install -y unzip zip gcc

# Navigate to the dictionaries directory
cd dictionary_learning
mkdir -p dictionaries
cd dictionaries

# HuggingFace repository name
REPO_NAME="canrager/lm_sae"

# List of files to download
FILES=(
    "pythia70m_sweep_standard_ctx128_0712.zip"
    "pythia70m_test_sae.zip"
    "pythia70m_sweep_topk_ctx128_0730.zip"
    "pythia70m_sweep_gated_ctx128_0730.zip"
    "all_730_results.zip"
)

# Download and extract each file
for file in "${FILES[@]}"; do
    download_and_extract "$REPO_NAME" "$file"
done

# Set up Triton kernels for Vast AI machines
export CC=/usr/bin/gcc
gcc --version
echo $CC