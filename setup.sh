pip3 install -r requirements.txt 
pip3 install torch torchvision torchaudio
pip3 install bitsandbytes
pip3 install --upgrade typing-extensions
# export HF_TOKEN=""
pip3 install 'accelerate>=0.26.0'
pip3 install protobuf
pip3 install sentencepiece
pip3 install pydantic
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip3 install torchao
