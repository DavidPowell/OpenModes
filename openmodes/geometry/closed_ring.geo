// a planar closed ring

If (!Exists(inner_radius))
    inner_radius = 3.5e-3;
EndIf

If (!Exists(outer_radius))
    outer_radius = 4e-3;
EndIf

If (!Exists(mesh_tol))
    mesh_tol = 2e-3;
EndIf

p = newp-1;

// define all points on the face
Point(p+1) = {0, 0, 0, mesh_tol};
Point(p+2) = {-inner_radius, 0, 0, mesh_tol};
Point(p+3) = {inner_radius, 0, 0, mesh_tol};
Point(p+4) = {-outer_radius, 0, 0, mesh_tol};
Point(p+5) = {outer_radius, 0, 0, mesh_tol};

l = newl-1;

// then the lines making the face
Circle(l+1) = {p+2, p+1, p+3};
Circle(l+2) = {p+3, p+1, p+2};

Circle(l+3) = {p+5, p+1, p+4};
Circle(l+4) = {p+4, p+1, p+5};

Line Loop(l+7) = {l+1, l+2};
Line Loop(l+8) = {l+3, l+4};

s = news;

Plane Surface(s) = {l+8, l+7};


