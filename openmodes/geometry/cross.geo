// a flat cross

If (!Exists(width))
    width = 1e-3;
EndIf

If (!Exists(height))
    height = 10e-3;
EndIf

If (!Exists(complementary_radius))
    complementary_radius = 0.0;
EndIf

If (!Exists(mesh_tol))
    mesh_tol=2e-3;
EndIf

p = newp-1;

// Define the dipole
// define all points on the face
Point(p+1) = {-0.5*width, -0.5*height, 0, mesh_tol};
Point(p+2) = { 0.5*width, -0.5*height, 0, mesh_tol};
Point(p+3) = { 0.5*width, -0.5*width, 0, mesh_tol};
Point(p+4) = {0.5*height, -0.5*width, 0, mesh_tol};
Point(p+5) = {0.5*height, 0.5*width, 0, mesh_tol};
Point(p+6) = {0.5*width, 0.5*width, 0, mesh_tol};
Point(p+7) = {0.5*width, 0.5*height, 0, mesh_tol};
Point(p+8) = {-0.5*width, 0.5*height, 0, mesh_tol};
Point(p+9) = {-0.5*width, 0.5*width, 0, mesh_tol};
Point(p+10) = {-0.5*height, 0.5*width, 0, mesh_tol};
Point(p+11) = {-0.5*height, -0.5*width, 0, mesh_tol};
Point(p+12) = {-0.5*width, -0.5*width, 0, mesh_tol};

l = newl-1;

Line(l+1) = {1, 2};
Line(l+2) = {2, 3};
Line(l+3) = {3, 4};
Line(l+4) = {4, 5};
Line(l+5) = {5, 6};
Line(l+6) = {6, 7};
Line(l+7) = {7, 8};
Line(l+8) = {8, 9};
Line(l+9) = {9, 10};
Line(l+10) = {10, 11};
Line(l+11) = {11, 12};
Line(l+12) = {12, 1};

ll = newll;
Line Loop(ll) = {l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9, l+10, l+11, l+12};

s = news;

If (complementary_radius == 0.0)
    Plane Surface(s) = {ll};
    Physical Surface(s+1) = {s};
EndIf

// Optionally, create the complementary structure, and embed it
// in a circle of given radius. Setting this radius to 0 (the default)
// means that the positive structure will be created.
If (complementary_radius != 0.0)
    p2 = newp-1;
    Point(p2+1) = {complementary_radius, 0, 0, mesh_tol};
    Point(p2+2) = {0, complementary_radius, 0, mesh_tol};
    Point(p2+3) = {-complementary_radius, 0, 0, mesh_tol};
    Point(p2+4) = {0, -complementary_radius, 0, mesh_tol};
    Point(p2+5) = {0, 0, 0, mesh_tol};
    
    l = newl-1;
    Circle(l+1) = {p2+1, p2+5, p2+2};
    Circle(l+2) = {p2+2, p2+5, p2+3};
    Circle(l+3) = {p2+3, p2+5, p2+4};
    Circle(l+4) = {p2+4, p2+5, p2+1};
    
    Line Loop(ll+1) = {l+1, l+2, l+3, l+4};
    
    Plane Surface(s) = {ll+1, ll};
    Physical Surface(s+1) = {s};
EndIf
