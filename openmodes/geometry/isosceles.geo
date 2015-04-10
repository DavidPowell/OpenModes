// an isosceles triangle

If (!Exists(width))
    width = 1e-3;
EndIf

If (!Exists(height))
    height = 10e-3;
EndIf

lc = 5e-3;

p = newp-1;

// Define the dipole
// define all points on the face
Point(p+1) = {-0.5*width, 0, 0, lc};
Point(p+2) = { 0.5*width, 0, 0, lc};
Point(p+3) = { 0,  height, 0, lc};

l = newl-1;

// then the lines making the face
Line(l+1) = {p+1, p+2};
Line(l+2) = {p+2, p+3};
Line(l+3) = {p+3, p+1};

Line Loop(l+5) = {l+1, l+2, l+3};

s = news;

Plane Surface(s) = {l+5};
