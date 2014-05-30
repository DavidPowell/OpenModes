// a single pair of triangles, representing one RWG basis function

width = 1e-3;
height = 2e-3;

lc = 2e-3;

p = newp-1;

// Define the SRR
// define all points on the face
Point(p+1) = {-0.5*width, 0, 0, lc};
Point(p+2) = { 0, -0.5*height, 0, lc};
Point(p+3) = { 0.5*width,  0, 0, lc};
Point(p+4) = {0,  0.5*height, 0, lc};

l = newl-1;

// then the lines making the face
Line(l+1) = {p+1, p+2};
Line(l+2) = {p+2, p+3};
Line(l+3) = {p+3, p+4};
Line(l+4) = {p+4, p+1};

Line Loop(l+5) = {l+1, l+2, l+3, l+4}; 

s = news;

Plane Surface(s) = {l+5};
