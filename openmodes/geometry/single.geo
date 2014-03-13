// a single pair of triangles, representing one RWG basis function

// first design the SRR
width = 1e-3;
height = 4e-3;
metal_thickness = 0; //0.03e-3;

//lc_srr = 0.5e-3;
lc_srr = 2e-3;

srr_p = newp-1;

// Define the SRR
// define all points on the face
Point(srr_p+1) = {-0.5*width, 0, 0, lc_srr};
Point(srr_p+2) = { 0, -0.5*height, 0, lc_srr};
Point(srr_p+3) = { 0.5*width,  0, 0, lc_srr};
Point(srr_p+4) = {0,  0.5*height, 0, lc_srr};

srr_l = newl-1;

// then the lines making the face
Line(srr_l+1) = {srr_p+1, srr_p+2};
Line(srr_l+2) = {srr_p+2, srr_p+3};
Line(srr_l+3) = {srr_p+3, srr_p+4};
Line(srr_l+4) = {srr_p+4, srr_p+1};

Line Loop(srr_l+5) = {srr_l+1, srr_l+2, srr_l+3, srr_l+4}; 

srr_s = news;

Plane Surface(srr_s) = {srr_l+5};
