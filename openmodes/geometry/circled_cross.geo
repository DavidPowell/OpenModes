// a cross with circular arcs on the ends of its arms, or optionally a gammadion
// specify the parameters positive_arc_angle and negative_arc_angle (in degrees) to control
// the circular arcs at the ends of the arms

If (width == 0.0)
    width = 1e-3;
EndIf

If (r_outer == 0.0)
    r_outer = 10e-3;
EndIf

negative_arc_angle = negative_arc_angle/180.0*Pi;
positive_arc_angle = positive_arc_angle/180.0*Pi;

lc = 2e-3;

p = newp-1;

r_inner = r_outer-width;

// Define the dipole
// define all points on the face
Point(p+1) = {0, 0, 0, lc};

Point(p+2) = {-0.5*width, 0.5*width, 0, lc};

y_inner = Sqrt(r_inner^2 -(0.5*width)^2);
y_outer = Sqrt(r_outer^2 -(0.5*width)^2);

Point(p+3) = {-0.5*width, y_inner, 0.0, lc};

If (negative_arc_angle != 0.0)
    Point(p+4) = {-r_inner*Sin(negative_arc_angle), r_inner*Cos(negative_arc_angle), 0.0, lc};
    Point(p+5) = {-r_outer*Sin(negative_arc_angle), r_outer*Cos(negative_arc_angle), 0.0, lc};
EndIf

If (negative_arc_angle == 0.0)
    Point(p+4) = {-0.5*width, 0.5*(y_inner+y_outer), 0.0, lc};
    Point(p+5) = {-0.5*width, y_outer, 0.0, lc};
EndIf

If (positive_arc_angle != 0.0)
    Point(p+6) = {r_outer*Sin(positive_arc_angle), r_outer*Cos(positive_arc_angle), 0.0, lc};
    Point(p+7) = {r_inner*Sin(positive_arc_angle), r_inner*Cos(positive_arc_angle), 0.0, lc};
EndIf

If (positive_arc_angle == 0.0)
    Point(p+6) = {0.5*width, y_outer, 0.0, lc};
    Point(p+7) = {0.5*width, 0.5*(y_inner+y_outer), 0.0, lc};
EndIf

Point(p+8) = {0.5*width, y_inner, 0.0, lc};

l = newl-1;

Line(l+1) = {p+2, p+3};

If (negative_arc_angle != 0.0)
    Circle(l+2) = {p+3, p+1, p+4};
EndIf
If (negative_arc_angle == 0.0)
    Line(l+2) = {p+3, p+4};
EndIf

Line(l+3) = {p+4, p+5};
Circle(l+4) = {p+5, p+1, p+6};
Line(l+5) = {p+6, p+7};

If (positive_arc_angle != 0.0)
    Circle(l+6) = {p+7, p+1, p+8};
EndIf
If (positive_arc_angle == 0.0)
    Line(l+6) = {p+7, p+8};
EndIf


copy1[] = Rotate {{0, 0, 1}, {0, 0, 0}, -0.5*Pi} {Duplicata {Point {p+2, p+3, p+4, p+5, p+6, p+7, p+8 }; Line {l+1, l+2, l+3, l+4, l+5, l+6 };}};
copy2[] = Rotate {{0, 0, 1}, {0, 0, 0}, Pi} {Duplicata {Point {p+2, p+3, p+4, p+5, p+6, p+7, p+8 }; Line {l+1, l+2, l+3, l+4, l+5, l+6 };}};
copy3[] = Rotate {{0, 0, 1}, {0, 0, 0}, 0.5*Pi} {Duplicata {Point {p+2, p+3, p+4, p+5, p+6, p+7, p+8 }; Line {l+1, l+2, l+3, l+4, l+5, l+6 };}};

l2 = newl-1;

Line(l2+1) = {p+8, copy1[0]};
Line(l2+2) = {copy1[6], copy2[0]};
Line(l2+3) = {copy2[6], copy3[0]};
Line(l2+4) = {copy3[6], l+2};

ll = newll;


Line Loop(ll) = {l+1, l+2, l+3, l+4, l+5, l+6, l2+1,
                       copy1[7], copy1[8], copy1[9], copy1[10], copy1[11], copy1[12], l2+2,
                       copy2[7], copy2[8], copy2[9], copy2[10], copy2[11], copy2[12], l2+3,
                       copy3[7], copy3[8], copy3[9], copy3[10], copy3[11], copy3[12], l2+4};

s = news;
Plane Surface(s) = {ll};
Physical Surface(s+1) = {s};

