// a box with sharp edges

// geometric parameters specifiable from the command line
If (!Exists(len_x))
    len_x = 12e-3;
EndIf

If (!Exists(len_y))
    len_y = 12e-3;
EndIf

If (!Exists(len_z))
    len_z = 3e-3;
EndIf

If (!Exists(mesh_tol))
    mesh_tol = 3e-3;
EndIf

p = newp-1;

// define all points on the bottom face
Point(p+1) = {-0.5*len_x, -0.5*len_y, -0.5*len_z, mesh_tol};
Point(p+2) = {-0.5*len_x,  0.5*len_y, -0.5*len_z, mesh_tol};
Point(p+3) = { 0.5*len_x,  0.5*len_y, -0.5*len_z, mesh_tol};
Point(p+4) = { 0.5*len_x, -0.5*len_y, -0.5*len_z, mesh_tol};


l = newl-1;

// then the lines making the face
Line(l+1) = {p+1, p+2};
Line(l+2) = {p+2, p+3};
Line(l+3) = {p+3, p+4};
Line(l+4) = {p+4, p+1};

Line Loop(l+9) = {l+1, l+2, l+3, l+4};

s = news;

// gmsh extrusian has problems with surface normals, which are fixed
// by using the following sequence of commands
Plane Surface(s) = {-(l+9)};
out[] = Extrude{0,0,len_z}{ Surface{s}; };
Reverse Surface{s};

