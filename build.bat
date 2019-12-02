@echo off
del build /q /s
del openmodes\*.pyd
python setup.py clean
python setup.py develop --no-deps

