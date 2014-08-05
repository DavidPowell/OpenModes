// a horseshoe without rounded edges

// geometric parameters specifiable from the command line
If (width == 0.0)
    width = 12e-3;
EndIf

If (length == 0.0)
    length = 12e-3;
EndIf

If (height == 0.0)
    height = 3e-3;
EndIf

If (track == 0.0)
    track = 3e-3;
EndIf

If (mesh_tol == 0.0)
    mesh_tol = 5e-3;
EndIf

p = newp-1;

// define all points on the bottom face
Point(p+1) = {-0.5*width, -0.5*length, 0, mesh_tol};
Point(p+2) = {-0.5*width,  0.5*length, 0, mesh_tol};
Point(p+3) = { 0.5*width,  0.5*length, 0, mesh_tol};
Point(p+4) = { 0.5*width, -0.5*length, 0, mesh_tol};
Point(p+5) = { 0.5*width-track, -0.5*length, 0, mesh_tol};
Point(p+6) = { 0.5*width-track, 0.5*length-track, 0, mesh_tol};
Point(p+7) = { -0.5*width+track, 0.5*length-track, 0, mesh_tol};
Point(p+8) = { -0.5*width+track, -0.5*length, 0, mesh_tol};

l = newl-1;

// then the lines making the face
Line(l+1) = {p+1, p+2};
Line(l+2) = {p+2, p+3};
Line(l+3) = {p+3, p+4};
Line(l+4) = {p+4, p+5};
Line(l+5) = {p+5, p+6};
Line(l+6) = {p+6, p+7};
Line(l+7) = {p+7, p+8};
Line(l+8) = {p+8, p+1};

Line Loop(l+9) = {l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8}; 

s = news;

// gmsh extrusian has problems with surface normals, which are fixed
// by using the following sequence of commands
Plane Surface(s) = {-(l+9)};
out[] = Extrude{0,0,height}{ Surface{s}; };
Reverse Surface{s};

