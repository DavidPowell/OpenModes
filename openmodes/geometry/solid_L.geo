// A solid L-shape extruded to a certain thickness

// geometric parameters specifiable from the command line
If (!Exists(hor_len))
    hor_len = 12e-3;
EndIf

If (!Exists(hor_width))
    hor_width = 4e-3;
EndIf

If (!Exists(vert_len))
    vert_len = 12e-3;
EndIf

If (!Exists(vert_width))
    vert_width = 6e-3;
EndIf

If (!Exists(thickness))
    thickness = 3e-3;
EndIf

If (!Exists(mesh_tol))
    mesh_tol = 3e-3;
EndIf

p = newp-1;

// define all points on the bottom face
Point(p+1) = {0, 0, 0, mesh_tol};
Point(p+2) = {0, vert_len, 0, mesh_tol};
Point(p+3) = {vert_width, vert_len, 0, mesh_tol};
Point(p+4) = {vert_width, hor_width, 0, mesh_tol};
Point(p+5) = {hor_len, hor_width, 0, mesh_tol};
Point(p+6) = {hor_len, 0, 0, mesh_tol};

l = newl-1;

// then the lines making the face
Line(l+1) = {p+1, p+2};
Line(l+2) = {p+2, p+3};
Line(l+3) = {p+3, p+4};
Line(l+4) = {p+4, p+5};
Line(l+5) = {p+5, p+6};
Line(l+6) = {p+6, p+1};

ll = newll;

Line Loop(ll) = {l+1, l+2, l+3, l+4, l+5, l+6};

s = news;

// gmsh extrusian has problems with surface normals, which are fixed
// by using the following sequence of commands
Plane Surface(s) = {-(ll)};
out[] = Extrude{0,0,thickness}{ Surface{s}; };
Reverse Surface{s};

