// a planar circle

If (outer_radius == 0.0)
    outer_radius = 4e-3;
EndIf

If (mesh_tol == 0.0)
    mesh_tol = 3e-3;
EndIf

p = newp-1;

// define all points on the face
Point(p+1) = {0, 0, 0, mesh_tol};
Point(p+4) = {-outer_radius, 0, 0, mesh_tol};
Point(p+5) = {outer_radius, 0, 0, mesh_tol};

l = newl-1;

// then the lines making the face
Circle(l+3) = {p+5, p+1, p+4};
Circle(l+4) = {p+4, p+1, p+5};

Line Loop(l+8) = {l+3, l+4};

s = news;

Plane Surface(s) = {l+8};


