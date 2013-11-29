@echo off
del build /q /s
del openmodes\*.pyd
python setup.py build_ext --inplace -c mingw32

