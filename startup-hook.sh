# Install Brevitas
python3 -m pip install brevitas

# Install finn-base, with inference cost estimation
pip install onnxoptimizer==0.2.6
pip install git+https://github.com/Xilinx/finn-base.git@feature/itu_competition_21#egg=finn-base[onnx]
