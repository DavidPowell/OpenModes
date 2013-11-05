// a 2D split ring resonator

// Use the "MeshAdapt" mesh algorithm which seems to generate the most consistent element sizes
// hence producing the most well-conditioned results
//Mesh.Algorithm = 1;

// first design the SRR
inner_radius = 3.5e-3;
outer_radius = 4e-3;
gap_width = 1e-3;
metal_thickness = 0; //0.03e-3;

//lc_srr = 0.5e-3;
lc_srr = 2e-3;

srr_p = newp-1;

// Define the SRR
// define all points on the face
Point(srr_p+1) = {0, 0, 0, lc_srr};
Point(srr_p+2) = {-Sqrt(inner_radius^2-(0.5*gap_width)^2), -0.5*gap_width, 0, lc_srr};
Point(srr_p+3) = {inner_radius, 0, 0, lc_srr};
Point(srr_p+4) = {-Sqrt(inner_radius^2-(0.5*gap_width)^2), 0.5*gap_width, 0, lc_srr};
Point(srr_p+5) = {-Sqrt(outer_radius^2-(0.5*gap_width)^2), 0.5*gap_width, 0, lc_srr};
Point(srr_p+6) = {outer_radius, 0, 0, lc_srr};
Point(srr_p+7) = {-Sqrt(outer_radius^2-(0.5*gap_width)^2), -0.5*gap_width, 0, lc_srr};

srr_l = newl-1;

// then the lines making the face
Circle(srr_l+1) = {srr_p+2, srr_p+1, srr_p+3};
Circle(srr_l+2) = {srr_p+3, srr_p+1, srr_p+4};
Line(srr_l+3) = {srr_p+4, srr_p+5};
Circle(srr_l+4) = {srr_p+5, srr_p+1, srr_p+6};
Circle(srr_l+5) = {srr_p+6, srr_p+1, srr_p+7};
Line(srr_l+6) = {srr_p+7, srr_p+2};

Line Loop(srr_l+7) = {srr_l+1, srr_l+2, srr_l+3, srr_l+4, srr_l+5, srr_l+6};

srr_s = news;

Plane Surface(srr_s) = {srr_l+7};
Field[1] = Threshold;
Field[1].LcMin = 0.5;
Field[1].LcMax = 0.5;
Field[1].LcMin = 0.1;


