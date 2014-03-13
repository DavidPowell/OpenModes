// a flat square plate

width = 10e-3;
height = 10e-3;
metal_thickness = 0;

lc_plate = 5e-3;

plate_p = newp-1;

// Define the SRR
// define all points on the face
Point(plate_p+1) = {-0.5*width, -0.5*height, 0, lc_plate};
Point(plate_p+2) = { 0.5*width, -0.5*height, 0, lc_plate};
Point(plate_p+3) = { 0.5*width,  0.5*height, 0, lc_plate};
Point(plate_p+4) = {-0.5*width,  0.5*height, 0, lc_plate};

plate_l = newl-1;

// then the lines making the face
Line(plate_l+1) = {plate_p+1, plate_p+2};
Line(plate_l+2) = {plate_p+2, plate_p+3};
Line(plate_l+3) = {plate_p+3, plate_p+4};
Line(plate_l+4) = {plate_p+4, plate_p+1};

Line Loop(plate_l+5) = {plate_l+1, plate_l+2, plate_l+3, plate_l+4}; 

plate_s = news;

Plane Surface(plate_s) = {plate_l+5};
