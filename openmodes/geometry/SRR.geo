// a 2D split ring resonator

// allow the geometric parameters to be specified on the command-line
If (inner_radius == 0.0)
    inner_radius = 3.5e-3;
EndIf

If (outer_radius == 0.0)
    outer_radius = 4e-3;
EndIf

If (gap_width == 0.0)
    gap_width = 1e-3;
EndIf

If (mesh_tol == 0.0)
    mesh_tol = 2e-3;
EndIf

srr_p = newp-1;

// Define the SRR
// define all points on the face
Point(srr_p+1) = {0, 0, 0, mesh_tol};
Point(srr_p+2) = {-Sqrt(inner_radius^2-(0.5*gap_width)^2), -0.5*gap_width, 0, mesh_tol};
Point(srr_p+3) = {inner_radius, 0, 0, mesh_tol};
Point(srr_p+4) = {-Sqrt(inner_radius^2-(0.5*gap_width)^2), 0.5*gap_width, 0, mesh_tol};
Point(srr_p+5) = {-Sqrt(outer_radius^2-(0.5*gap_width)^2), 0.5*gap_width, 0, mesh_tol};
Point(srr_p+6) = {outer_radius, 0, 0, mesh_tol};
Point(srr_p+7) = {-Sqrt(outer_radius^2-(0.5*gap_width)^2), -0.5*gap_width, 0, mesh_tol};

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



