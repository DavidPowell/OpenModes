lc = 2e-3;

If (!Exists(inner_radius))
    inner_radius = 3e-3;
EndIf

If (!Exists(outer_radius))
    outer_radius = 4e-3;
EndIf

If (!Exists(gap_width))
    gap_width = 3e-3;
EndIf

If (!Exists(arm_length))
    arm_length = 10e-3;
EndIf

gap_inner_x = -Sqrt(inner_radius^2-(0.5*gap_width)^2);
gap_outer_x = -Sqrt(outer_radius^2-(0.5*gap_width)^2);
gap_r = 0.5*gap_width;

p = newp-1;
l = newl-1;
loop_l = newll;
s = news-1;

// The loop face
Point(p+1) = {0, 0, 0, lc};
Point(p+2) = {gap_inner_x, -gap_r, 0, lc};
Point(p+3) = {inner_radius, 0, 0, lc};
Point(p+4) = {gap_inner_x, gap_r, 0, lc};
Point(p+5) = {gap_outer_x, gap_r, 0, lc};
Point(p+6) = {outer_radius, 0, 0, lc};
Point(p+7) = {gap_outer_x, -gap_r, 0, lc};

Circle(l+1) = {p+2, p+1, p+3};
Circle(l+2) = {p+3, p+1, p+4};
Line(l+3) = {p+4, p+5};
Circle(l+4) = {p+5, p+1, p+6};
Circle(l+5) = {p+6, p+1, p+7};
Line(l+6) = {p+7, p+2};

Line Loop(loop_l+1) = {l+1, l+2, l+3, l+4, l+5, l+6};
Plane Surface(s+1) = {loop_l+1};

// The curved lower join
Point(p+8) = {gap_outer_x, -gap_r, -gap_r, lc};
Point(p+9) = {gap_outer_x, 0, -gap_r, lc};
Point(p+10) = {gap_inner_x, 0, -gap_r, lc};
Point(p+11) = {gap_inner_x, -gap_r, -gap_r, lc};

Circle(l+7) = {p+7, p+8, p+9};
Circle(l+11) = {p+10, p+11, p+2};
Line(l+12) = {p+9, p+10};

Line Loop(loop_l+2) = {-(l+6), l+7, l+12, l+11};
Ruled Surface(s+2) = {loop_l+2};

// The lower arm
Point(p+12) = {gap_outer_x, 0, -arm_length, lc};
Point(p+13) = {gap_inner_x, 0, -arm_length, lc};
Point(p+14) = {gap_outer_x, 0, -arm_length, lc};
Point(p+15) = {gap_inner_x, 0, -arm_length, lc};

Line(l+8) = {p+9, p+12};
Line(l+9) = {p+12, p+13};
Line(l+10) = {p+13, p+10};

Line Loop(loop_l+3) = {l+8, l+9, l+10, -(l+12)};
Plane Surface(s+3) = {loop_l+3};

// The curved upper join
Point(p+16) = {gap_outer_x, gap_r, gap_r, lc};
Point(p+17) = {gap_outer_x, 0, gap_r, lc};
Point(p+18) = {gap_inner_x, 0, gap_r, lc};
Point(p+19) = {gap_inner_x, gap_r, gap_r, lc};

Circle(l+13) = {p+5, p+16, p+17};
Line(l+14) = {p+17, p+18};
Circle(l+15) = {p+18, p+19, p+4};

Line Loop(loop_l+4) = {l+13, l+14, l+15, l+3};
Ruled Surface(s+4) = {loop_l+4};

// The upper arm
Point(p+20) = {gap_outer_x, 0, arm_length, lc};
Point(p+21) = {gap_inner_x, 0, arm_length, lc};
Point(p+22) = {gap_outer_x, 0, arm_length, lc};
Point(p+23) = {gap_inner_x, 0, arm_length, lc};

Line(l+16) = {p+17, p+20};
Line(l+17) = {p+20, p+21};
Line(l+18) = {p+21, p+18};

Line Loop(loop_l+5) = {l+16, l+17, l+18, -(l+14)};
Plane Surface(s+5) = {loop_l+5};

Compound Surface(s+6) = {s+1, s+2, s+3, s+4, s+5, s+6};
Physical Surface(s+7) = {s+6};
