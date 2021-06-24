# Dependencies

There might be issues with torchtext, since it was changing a lot during this course.

Please use `PyTorch 1.9.0` and `TorchText 0.9.0` for all of the notebooks to work.

You can also try `pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html` which might fix, but again this is nightly and it changes a lot as well.

## UPDATE:

This fixed it:

```
! pip install torch==1.8.1+cu102 torchtext==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
