# Install Brevitas
python3 -m pip install brevitas==0.6.0
# Install finn-base, with inference cost estimation
python3 -m pip install onnxoptimizer==0.2.6
python3 -m pip install git+https://github.com/Xilinx/finn-base.git@feature/itu_competition_21#egg=finn-base[onnx]
