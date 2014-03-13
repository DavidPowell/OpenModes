// An asymmetric split ring, as per Fedotov

// first design the SRR
outer_radius = 6e-3;
inner_radius = 5.2e-3;

angle1 = 0.5*140.00*Pi/180;
angle2 =  0.5*160.00*Pi/180;

lc = 2e-3;

p = newp-1;

Point(p+1) = {0, 0, 0, lc};
Point(p+2) = {inner_radius*Sin(angle1), inner_radius*Cos(angle1), 0, lc};
Point(p+3) = {-inner_radius*Sin(angle1), inner_radius*Cos(angle1), 0, lc};
Point(p+4) = {-outer_radius*Sin(angle1), outer_radius*Cos(angle1), 0, lc};
Point(p+5) = {outer_radius*Sin(angle1), outer_radius*Cos(angle1), 0, lc};

l = newl-1;

Circle(l+1) = {p+2, p+1, p+3};
Line(l+2) = {p+3, p+4};
Circle(l+3) = {p+4, p+1, p+5};
Line(l+4) = {p+5, p+2};

Line Loop(l+5) = {l+1, l+2, l+3, l+4};

s = news-1;

Plane Surface(s+1) = {l+5};

Point(p+6) = {inner_radius*Sin(angle2), -inner_radius*Cos(angle2), 0, lc};
Point(p+7) = {-inner_radius*Sin(angle2), -inner_radius*Cos(angle2), 0, lc};
Point(p+8) = {-outer_radius*Sin(angle2), -outer_radius*Cos(angle2), 0, lc};
Point(p+9) = {outer_radius*Sin(angle2), -outer_radius*Cos(angle2), 0, lc};

Circle(l+6) = {p+6, p+1, p+7};
Line(l+7) = {p+7, p+8};
Circle(l+8) = {p+8, p+1, p+9};
Line(l+9) = {p+9, p+6};

Line Loop(l+10) = {l+6, l+7, l+8, l+9};

Plane Surface(s+2) = {l+10};

Physical Surface("left") = {s+1};
Physical Surface("right") = {s+2};

