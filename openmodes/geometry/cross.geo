// a flat cross

If (width == 0.0)
    width = 1e-3;
EndIf

If (height == 0.0)
    height = 10e-3;
EndIf

lc = 2e-3;

p = newp-1;

// Define the dipole
// define all points on the face
Point(p+1) = {-0.5*width, -0.5*height, 0, lc};
Point(p+2) = { 0.5*width, -0.5*height, 0, lc};
Point(p+3) = { 0.5*width, -0.5*width, 0, lc};
Point(p+4) = {0.5*height, -0.5*width, 0, lc};
Point(p+5) = {0.5*height, 0.5*width, 0, lc};
Point(p+6) = {0.5*width, 0.5*width, 0, lc};
Point(p+7) = {0.5*width, 0.5*height, 0, lc};
Point(p+8) = {-0.5*width, 0.5*height, 0, lc};
Point(p+9) = {-0.5*width, 0.5*width, 0, lc};
Point(p+10) = {-0.5*height, 0.5*width, 0, lc};
Point(p+11) = {-0.5*height, -0.5*width, 0, lc};
Point(p+12) = {-0.5*width, -0.5*width, 0, lc};

l = newl-1;

Line(l+1) = {1, 2};
Line(l+2) = {2, 3};
Line(l+3) = {3, 4};
Line(l+4) = {4, 5};
Line(l+5) = {5, 6};
Line(l+6) = {6, 7};
Line(l+7) = {7, 8};
Line(l+8) = {8, 9};
Line(l+9) = {9, 10};
Line(l+10) = {10, 11};
Line(l+11) = {11, 12};
Line(l+12) = {12, 1};

ll = newll;
Line Loop(ll) = {l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9, l+10, l+11, l+12};

s = news;
Plane Surface(s) = {ll};
