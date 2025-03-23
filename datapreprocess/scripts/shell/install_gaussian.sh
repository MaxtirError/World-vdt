mkdir -p /tmp/extensions
git clone https://github.com/graphdeco-inria/gaussian-splatting /tmp/extensions/gaussian-splatting --recursive
pip install /tmp/extensions/gaussian-splatting/submodules/diff-gaussian-rasterization/
