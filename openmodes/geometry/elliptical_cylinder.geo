// an elliptical cylinder with rounded edges

// radius in x dimension
If (!Exists(radius_x))
    radius_x = 4e-3;
EndIf

// radius in y dimension
If (!Exists(radius_y))
    radius_y = 6e-3;
EndIf

// radius of rounding
If (!Exists(rounding))
    rounding = radius_x*0.2;
EndIf

// total height
If (!Exists(height))
    height = 4e-3;
EndIf

If (!Exists(mesh_tol))
    mesh_tol = radius_x/3.0;
EndIf

p = newp-1;
l = newl-1;

end_radius_x = radius_x-rounding;
end_radius_y = radius_y-rounding;

// The bottom face
bottom_z = -height*0.5;

Point(p+1) = {0, 0, bottom_z, mesh_tol};
Point(p+2) = {-end_radius_x, 0, bottom_z, mesh_tol};
Point(p+3) = {0, end_radius_y, bottom_z, mesh_tol};
Point(p+4) = {end_radius_x, 0, bottom_z, mesh_tol};
Point(p+5) = {0, -end_radius_y, bottom_z, mesh_tol};

// determine a point on the major axis
If (radius_x > radius_y)
    major_point = p+2;
Else
    major_point = p+3;
EndIf

// start, centre, point on major axis, end
Ellipse(l+1) = {p+2, p+1, major_point, p+3};
Ellipse(l+2) = {p+3, p+1, major_point, p+4};
Ellipse(l+3) = {p+4, p+1, major_point, p+5};
Ellipse(l+4) = {p+5, p+1, major_point, p+2};

// The bottom full width circle
lower_z  = -height*0.5+rounding;

Point(p+6) = {0, 0, lower_z, mesh_tol};
Point(p+7) = {-radius_x, 0, lower_z, mesh_tol};
Point(p+8) = {0, radius_y, lower_z, mesh_tol};
Point(p+9) = {radius_x, 0, lower_z, mesh_tol};
Point(p+10) = {0, -radius_y, lower_z, mesh_tol};

// determine a point on the major axis
If (radius_x > radius_y)
    major_point = p+7;
Else
    major_point = p+8;
EndIf

Ellipse(l+5) = {p+7, p+6, major_point, p+8};
Ellipse(l+6) = {p+8, p+6, major_point, p+9};
Ellipse(l+7) = {p+9, p+6, major_point, p+10};
Ellipse(l+8) = {p+10, p+6, major_point, p+7};


// The top full width circle
upper_z  = height*0.5-rounding;

Point(p+11) = {0, 0, upper_z, mesh_tol};
Point(p+12) = {-radius_x, 0, upper_z, mesh_tol};
Point(p+13) = {0, radius_y, upper_z, mesh_tol};
Point(p+14) = {radius_x, 0, upper_z, mesh_tol};
Point(p+15) = {0, -radius_y, upper_z, mesh_tol};

// determine a point on the major axis
If (radius_x > radius_y)
    major_point = p+12;
Else
    major_point = p+13;
EndIf

Ellipse(l+9) = {p+12, p+11, major_point, p+13};
Ellipse(l+10) = {p+13, p+11, major_point, p+14};
Ellipse(l+11) = {p+14, p+11, major_point, p+15};
Ellipse(l+12) = {p+15, p+11, major_point, p+12};


// The top face
top_z = height*0.5;

Point(p+16) = {0, 0, top_z, mesh_tol};
Point(p+17) = {-end_radius_x, 0, top_z, mesh_tol};
Point(p+18) = {0, end_radius_y, top_z, mesh_tol};
Point(p+19) = {end_radius_x, 0, top_z, mesh_tol};
Point(p+20) = {0, -end_radius_y, top_z, mesh_tol};

// determine a point on the major axis
If (radius_x > radius_y)
    major_point = p+17;
Else
    major_point = p+18;
EndIf

Ellipse(l+13) = {p+17, p+16, major_point, p+18};
Ellipse(l+14) = {p+18, p+16, major_point, p+19};
Ellipse(l+15) = {p+19, p+16, major_point, p+20};
Ellipse(l+16) = {p+20, p+16, major_point, p+17};


// The bottom rounded sections

Point(p+21) = {-end_radius_x, 0, lower_z, mesh_tol};
Point(p+22) = {0, end_radius_y, lower_z, mesh_tol};
Point(p+23) = {end_radius_x, 0, lower_z, mesh_tol};
Point(p+24) = {0, -end_radius_y, lower_z, mesh_tol};

Circle(l+17) = {p+2, p+21, p+7};
Circle(l+18) = {p+3, p+22, p+8};
Circle(l+19) = {p+4, p+23, p+9};
Circle(l+20) = {p+5, p+24, p+10};

// The top rounded sections

Point(p+25) = {-end_radius_x, 0, upper_z, mesh_tol};
Point(p+26) = {0, end_radius_y, upper_z, mesh_tol};
Point(p+27) = {end_radius_x, 0, upper_z, mesh_tol};
Point(p+28) = {0, -end_radius_y, upper_z, mesh_tol};

Circle(l+21) = {12, 25, 17};
Circle(l+22) = {13, 26, 18};
Circle(l+23) = {14, 27, 19};
Circle(l+24) = {15, 28, 20};

// The sides

Line(l+25) = {p+7, p+12};
Line(l+26) = {p+8, p+13};
Line(l+27) = {p+9, p+14};
Line(l+28) = {p+10, p+15};

// Form the surfaces
s = news-1;

// top and bottom
Line Loop(l+100) = {l+1, l+2, l+3, l+4};
Plane Surface(s+1) = {l+100};
Line Loop(l+101) = {l+13, l+14, l+15, l+16};
Plane Surface(s+2) = {-(l+101)};

// sides
Line Loop(l+102) = {l+5, -(l+25), -(l+9), (l+26)};
Ruled Surface(s+3) = {-(l+102)};
Line Loop(l+103) = {l+6, -(l+26), -(l+10), (l+27)};
Ruled Surface(s+4) = {-(l+103)};
Line Loop(l+104) = {l+7, -(l+27), -(l+11), (l+28)};
Ruled Surface(s+5) = {-(l+104)};
Line Loop(l+105) = {l+8, -(l+28), -(l+12), (l+25)};
Ruled Surface(s+6) = {-(l+105)};

// bottom rounding
Line Loop(l+106) = {l+1, -(l+17), -(l+5), (l+18)};
Ruled Surface(s+7) = {-(l+106)};
Line Loop(l+107) = {l+2, -(l+18), -(l+6), (l+19)};
Ruled Surface(s+8) = {-(l+107)};
Line Loop(l+108) = {l+3, -(l+19), -(l+7), (l+20)};
Ruled Surface(s+9) = {-(l+108)};
Line Loop(l+109) = {l+4, -(l+20), -(l+8), (l+17)};
Ruled Surface(s+10) = {-(l+109)};

// top rounding
Line Loop(l+110) = {l+9, -(l+21), -(l+13), (l+22)};
Ruled Surface(s+11) = {-(l+110)};
Line Loop(l+111) = {l+10, -(l+22), -(l+14), (l+23)};
Ruled Surface(s+12) = {-(l+111)};
Line Loop(l+112) = {l+11, -(l+23), -(l+15), (l+24)};
Ruled Surface(s+13) = {-(l+112)};
Line Loop(l+113) = {l+12, -(l+24), -(l+16), (l+21)};
Ruled Surface(s+14) = {-(l+113)};

// Close the surface
Surface Loop(s+50) = {s+1, s+2, s+3, s+4, s+5, s+6, s+7, s+8, s+9, s+10, s+11, s+12, s+13, s+14};
Volume(1) = {s+50};
