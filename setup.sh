pip install -r requirements.txt
pip install -e .
git submodule update --init

apt install unzip
apt install zip

cd dictionary_learning
mkdir dictionaries
cd dictionaries

wget -O pythia70m_sweep_standard_ctx128_0712.zip "https://huggingface.co/canrager/lm_sae/resolve/main/pythia70m_sweep_standard_ctx128_0712.zip?download=true"
unzip pythia70m_sweep_standard_ctx128_0712.zip


wget -O pythia70m_test_sae.zip "https://huggingface.co/canrager/lm_sae/resolve/main/pythia70m_test_sae.zip?download=true"
unzip pythia70m_test_sae.zip



# For Triton kernels on Vast AI machines
sudo apt-get update
sudo apt-get install gcc
export CC=/usr/bin/gcc
gcc --version
echo $CC
