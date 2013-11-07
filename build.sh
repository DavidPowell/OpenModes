#f2py -c dunavant.f90 -m dunavant
#f2py -c core_for.f90 fpbspl.f splev.f -m core_for --f90flags="-g -pg" -lgomp only: interpolate_greens_inplane face_integrals_hanninen face_integrals_interpolated impedance_core_interpolated impedance_core_hanninen
#-fopenmp -fbounds-check -fimplicit-none 
# -DF2PY_REPORT_ON_ARRAY_COPY=1 
rm -Rf build
python setup.py build_ext --inplace

