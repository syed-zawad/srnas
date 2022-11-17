# Code for DySR.

# Work in progress, not yet suitable for third-parties/public

**Removed all proprietary and indentifiable code. Nothing is linked together in current version**

To check a sample of one-shot training, see train.py
To see a simple sample on how dynamic networks works, see model.py and train.py. Currently only works with RDDN/RDN blocks. More updates coming…

To Do -
  1. Replace all proprietary code with public versions.
    a) First need to link via re-implementing missing modules - **PRIORITY**
    b) Need to add back ms_ and upsampler modules
    c) Check comments and ensure all proprietary and identifiable code is removed
 2. Correctly link to third-party repositories.
    a) Currently contains many other open-source additions, fix it and properly link/cite them
    b) Removed ONNX build, use open source version now.
 3. Replace #### shell scripts.
    a) Once everything is working, replace these with generic bash scripts for easy runs
    b) Make sure no identifiable code or comments
    c) How to replace #####? Find a way without increasing deployment/profiling costs.
