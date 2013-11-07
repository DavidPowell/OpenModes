del build /q /s
rm -Rf build
python setup.py build_ext --inplace -c mingw32

