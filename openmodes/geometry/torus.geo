// A torus

// Allow radius to be specified on the command line
If (!Exists(major_radius))
    major_radius = 10e-3;
EndIf

If (!Exists(minor_radius))
    minor_radius = 3e-3;
EndIf

// base element size on radius
lc = minor_radius;

Point(1) = {major_radius,0.0,0.0,lc};
Point(2) = {major_radius+minor_radius, 0.0, 0.0, lc};
Point(3) = {major_radius, 0.0, minor_radius, lc};
Point(4) = {major_radius-minor_radius, 0.0, 0.0, lc};
Point(5) = {major_radius, 0.0, -minor_radius, lc};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(1) = {5};

Extrude {{0, 0, 1}, {0, 0, 0}, 2*Pi/3} {
  Surface{1};
}

Extrude {{0, 0, 1}, {0, 0, 0}, 2*Pi/3} {
  Surface{27};
}

Extrude {{0, 0, 1}, {0, 0, 0}, 2*Pi/3} {
  Surface{49};
}

Compound Volume(4) = {1, 2, 3};

// delete elements which aren't needed to prevent
// meshing of internal surfaces
Delete{ Volume{1, 2, 3};}
Delete{ Surface{1, 27, 49};}
