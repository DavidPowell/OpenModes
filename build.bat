del build /q /s
del *.pyd
rem rm -Rf build
python setup.py build_ext --inplace -c mingw32

