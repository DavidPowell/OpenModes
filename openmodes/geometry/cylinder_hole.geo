// A cylinder with rounded edges, having a partial hole through one side
// See Alaee et al, Phys Rev B 92, 245130 (2015)

// outer radius
If (!Exists(radius))
    radius = 4e-3;
EndIf

// radius of rounding
If (!Exists(rounding))
    rounding = radius*0.2;
EndIf

// total height
If (!Exists(height))
    height = 4e-3;
EndIf

// hole radius
If (!Exists(hole_radius))
    hole_radius = radius*0.3;
EndIf

// hole height
If (!Exists(hole_height))
    hole_height = height*0.5;
EndIf

If (!Exists(mesh_tol))
    mesh_tol = radius/3.0;
EndIf


p = newp-1;
l = newl-1;


bottom_z = -height*0.5;
top_z = height*0.5;

Point(p+1) = { 0, 0, bottom_z, mesh_tol };
Point(p+2) = { radius-rounding, 0, bottom_z, mesh_tol };
Point(p+3) = { radius-rounding, 0, bottom_z+rounding, mesh_tol };
Point(p+4) = { radius, 0, bottom_z+rounding, mesh_tol };
Point(p+5) = { radius, 0, top_z-rounding, mesh_tol };
Point(p+6) = { radius-rounding, 0, top_z-rounding, mesh_tol };
Point(p+7) = { radius-rounding, 0, top_z, mesh_tol };
Point(p+8) = { hole_radius+rounding, 0, top_z, mesh_tol };
Point(p+9) = { hole_radius+rounding, 0, top_z-rounding, mesh_tol };
Point(p+10) = { hole_radius, 0, top_z-rounding, mesh_tol };
Point(p+11) = { hole_radius, 0, top_z-hole_height+rounding, mesh_tol };
Point(p+12) = { hole_radius-rounding, 0, top_z-hole_height+rounding, mesh_tol };
Point(p+13) = { hole_radius-rounding, 0, top_z-hole_height, mesh_tol };
Point(p+14) = {0, 0, top_z-hole_height, mesh_tol };

Line(l+1) = { p+1, p+2 };
Circle(l+2) = {p+2, p+3, p+4};
Line(l+3) = {p+4, p+5};
Circle(l+4) = {p+5, p+6, p+7};
Line(l+5) = {p+7, p+8};
Circle(l+6) = {p+8, p+9, p+10};
Line(l+7) = {p+10, p+11};
Circle(l+8) = {p+11, p+12, p+13};
Line(l+9) = {p+13, p+14};

// The list "sides" will contain all the surfaces of the final object
sides[] = {};
For it In {1:9}
    rotated_geo[] = Extrude{ {0, 0, 1}, { 0, 0, 0 }, 0.5*Pi} {Line{l+it};};
    sides += rotated_geo[1];

    rotated_geo[] = Extrude{ {0, 0, 1}, { 0, 0, 0 }, 0.5*Pi} {Line{rotated_geo[0]};};
    sides += rotated_geo[1];

    rotated_geo[] = Extrude{ {0, 0, 1}, { 0, 0, 0 }, 0.5*Pi} {Line{rotated_geo[0]};};
    sides += rotated_geo[1];

    rotated_geo[] = Extrude{ {0, 0, 1}, { 0, 0, 0 }, 0.5*Pi} {Line{rotated_geo[0]};};
    sides += rotated_geo[1];
    
EndFor

