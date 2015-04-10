// a flat rectangle

If (!Exists(width))
    width = 1e-3;
EndIf

If (!Exists(height))
    height = 10e-3;
EndIf

If (!Exists(mesh_tol))
    mesh_tol = 3e-3;
EndIf

p = newp-1;

// Define the dipole
// define all points on the face
Point(p+1) = {-0.5*width, -0.5*height, 0, mesh_tol};
Point(p+2) = { 0.5*width, -0.5*height, 0, mesh_tol};
Point(p+3) = { 0.5*width,  0.5*height, 0, mesh_tol};
Point(p+4) = {-0.5*width,  0.5*height, 0, mesh_tol};

l = newl-1;

// then the lines making the face
Line(l+1) = {p+1, p+2};
Line(l+2) = {p+2, p+3};
Line(l+3) = {p+3, p+4};
Line(l+4) = {p+4, p+1};

Line Loop(l+5) = {l+1, l+2, l+3, l+4}; 

s = news;

Plane Surface(s) = {l+5};
