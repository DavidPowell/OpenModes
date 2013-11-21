@echo off
rem del build /q /s
rem del *.pyd
python setup.py build_ext --inplace -c mingw32

