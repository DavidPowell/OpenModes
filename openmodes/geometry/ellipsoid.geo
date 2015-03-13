// An ellipsoid sphere

// Allow the radii to be specified on the command line
If (radius_x == 0.0)
    radius_x = 10e-3;
EndIf

If (radius_y == 0.0)
    radius_y = 12e-3;
EndIf

If (radius_z == 0.0)
    radius_z = 8e-3;
EndIf

// base element size on radius
If (mesh_tol == 0.0)
    mesh_tol = radius*0.2;
EndIf

Point(1) = {0.0,0.0,0.0,mesh_tol};
Point(2) = {radius_x,0.0,0.0,mesh_tol};
Point(3) = {0,radius_y,0.0,mesh_tol};
Point(4) = {-radius_x,0,0.0,mesh_tol};
Point(5) = {0,-radius_y,0.0,mesh_tol};
Point(6) = {0,0,-radius_z,mesh_tol};
Point(7) = {0,0,radius_z,mesh_tol};

// Find the points on the major axis for each ellipse
If (radius_x > radius_y)
    major_xy = 2;
EndIf
If (radius_x <= radius_y)
    major_xy = 3;
EndIf

If (radius_x > radius_z)
    major_xz = 2;
EndIf
If (radius_x <= radius_z)
    major_xz = 6;
EndIf

If (radius_y > radius_z)
    major_yz = 3;
EndIf
If (radius_y <= radius_z)
    major_yz = 6;
EndIf

Ellipse(1) = {2,1,major_xy,3};
Ellipse(2) = {3,1,major_xy,4};
Ellipse(3) = {4,1,major_xy,5};
Ellipse(4) = {5,1,major_xy,2};
Ellipse(5) = {3,1,major_yz,6};
Ellipse(6) = {6,1,major_yz,5};
Ellipse(7) = {5,1,major_yz,7};
Ellipse(8) = {7,1,major_yz,3};
Ellipse(9) = {2,1,major_xz,7};
Ellipse(10) = {7,1,major_xz,4};
Ellipse(11) = {4,1,major_xz,6};
Ellipse(12) = {6,1,major_xz,2};

Line Loop(13) = {2,8,-10};
Ruled Surface(14) = {13};
Line Loop(15) = {10,3,7};
Ruled Surface(16) = {15};
Line Loop(17) = {-8,-9,1};
Ruled Surface(18) = {17};
Line Loop(19) = {-11,-2,5};
Ruled Surface(20) = {19};
Line Loop(21) = {-5,-12,-1};
Ruled Surface(22) = {21};
Line Loop(23) = {-3,11,6};
Ruled Surface(24) = {23};
Line Loop(25) = {-7,4,9};
Ruled Surface(26) = {25};
Line Loop(27) = {-4,12,-6};
Ruled Surface(28) = {27};
Surface Loop(29) = {28,26,16,14,20,24,22,18};

Volume(30) = {29};

